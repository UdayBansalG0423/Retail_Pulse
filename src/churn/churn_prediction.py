"""
churn_prediction.py
-------------------
Trains an XGBoost classifier to predict customer churn using historical transactions
and explains model predictions with SHAP (SHapley Additive exPlanations).

Methodology & Problem Formulation
---------------------------------
* Observation Cutoff : 2011-09-30 (features engineered strictly prior to cutoff)
* Churn Label Window : 2011-10-01 to 2011-11-29 (60 days)
* Churn Definition   : churn = 1 if customer has NO recorded transactions in the
                       60-day window following cutoff; churn = 0 if customer returns.
* Features           : RFM metrics, average order value, units/order, product diversity,
                       purchase span, tenure, and recent 30-day spending trends.
* Evaluation         : Stratified 75/25 train/test split. Reports ROC-AUC, Average
                       Precision (PR-AUC), and Precision@top-20%.

Outputs:
  - data/processed/churn_customer_predictions.csv
  - reports/churn/churn_metrics.json
  - reports/churn/churn_shap_importance.csv
  - reports/churn/churn_shap_summary.png
  - models/churn_xgb.json
"""

from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)
from xgboost import XGBClassifier
import shap

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "transactions_clean.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "churn"

PREDICTIONS_CSV = OUTPUT_DIR / "churn_customer_predictions.csv"
METRICS_JSON = REPORT_DIR / "churn_metrics.json"
SHAP_IMPORTANCE_CSV = REPORT_DIR / "churn_shap_importance.csv"
SHAP_SUMMARY_PNG = REPORT_DIR / "churn_shap_summary.png"
MODEL_JSON = MODEL_DIR / "churn_xgb.json"

CUTOFF_DATE = pd.Timestamp("2011-09-30 23:59:59")
LABEL_WINDOW_DAYS = 60
LABEL_END_DATE = CUTOFF_DATE + pd.Timedelta(days=LABEL_WINDOW_DAYS)
RANDOM_STATE = 42

FEATURES = [
    "recency_days",
    "frequency_dates",
    "total_spend",
    "total_units",
    "avg_order_value",
    "avg_units_per_order",
    "avg_spend_per_item",
    "unique_products",
    "tenure_days",
    "purchase_span_days",
    "recent_30d_spend",
    "recent_30d_orders",
    "spend_trend",
]


def load_transactions() -> pd.DataFrame:
    """Load cleaned transaction records with date parsing."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cleaned transactions file not found at: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Drop rows without CustomerID or with non-positive values
    df = df.dropna(subset=["CustomerID", "InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    return df


def build_churn_dataset(transactions: pd.DataFrame) -> pd.DataFrame:
    """Engineer historical features prior to cutoff and assign 60-day future churn labels."""
    hist = transactions[transactions["InvoiceDate"] <= CUTOFF_DATE].copy()
    future = transactions[
        (transactions["InvoiceDate"] > CUTOFF_DATE) &
        (transactions["InvoiceDate"] <= LABEL_END_DATE)
    ].copy()

    if hist.empty:
        raise ValueError("No historical transactions found prior to cutoff date.")

    # Identified returning customers in future 60-day window
    returning_customers = set(future["CustomerID"].unique())

    hist["Date"] = hist["InvoiceDate"].dt.date
    grouped = hist.groupby("CustomerID")

    features = grouped.agg(
        last_date=("InvoiceDate", "max"),
        first_date=("InvoiceDate", "min"),
        frequency_dates=("Date", "nunique"),
        total_spend=("Revenue", "sum"),
        total_units=("Quantity", "sum"),
        unique_products=("Description", "nunique"),
    )

    # Behavioral RFM & basket metrics
    features["recency_days"] = (CUTOFF_DATE - features["last_date"]).dt.days
    features["tenure_days"] = (CUTOFF_DATE - features["first_date"]).dt.days
    features["purchase_span_days"] = (features["last_date"] - features["first_date"]).dt.days
    features["avg_order_value"] = features["total_spend"] / features["frequency_dates"]
    features["avg_units_per_order"] = features["total_units"] / features["frequency_dates"]
    features["avg_spend_per_item"] = features["total_spend"] / features["total_units"]

    # Recent 30-day momentum prior to cutoff
    recent_start = CUTOFF_DATE - pd.Timedelta(days=30)
    recent_subset = hist[hist["InvoiceDate"] >= recent_start]
    recent_agg = recent_subset.groupby("CustomerID").agg(
        recent_30d_spend=("Revenue", "sum"),
        recent_30d_orders=("Date", "nunique"),
    )
    features = features.join(recent_agg).fillna({"recent_30d_spend": 0.0, "recent_30d_orders": 0})
    features["spend_trend"] = features["recent_30d_spend"] / (features["avg_order_value"] + 1e-6)

    # Ground truth churn target: 1 = no future transaction in 60d; 0 = active return
    features["churn"] = (~features.index.isin(returning_customers)).astype(int)

    features = features.reset_index()
    features["CustomerID"] = features["CustomerID"].astype(int)
    features = features[["CustomerID", *FEATURES, "churn"]]
    features[FEATURES] = features[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    if features["churn"].nunique() < 2:
        raise ValueError(
            "The chosen 60-day definition produced only one churn class. "
            "Inspect the label period before fitting a model."
        )

    return features


def precision_at_top_fraction(y_true, probabilities, fraction=0.20) -> float:
    """Precision among the highest-risk fraction of test customers."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    top_count = max(1, math.ceil(len(y_true) * fraction))
    ranked_indices = np.argsort(probabilities)[::-1][:top_count]
    return float(y_true[ranked_indices].mean())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RetailPulse — Customer Churn Prediction (XGBoost + SHAP)")
    print("=" * 70)
    print(f"Observation Cutoff Date : {CUTOFF_DATE.date()}")
    print(f"60-Day Evaluation Window: {CUTOFF_DATE.date() + pd.Timedelta(days=1)} to {LABEL_END_DATE.date()}")

    # 1. Load data and engineer features
    transactions = load_transactions()
    dataset = build_churn_dataset(transactions)
    n_total = len(dataset)
    n_churn = dataset["churn"].sum()
    churn_rate = n_churn / n_total

    print(f"\nCohort Size (Active Pre-Cutoff): {n_total:,} customers")
    print(f"Churned Customers (No 60d Return): {n_churn:,} ({churn_rate:.2%})")
    print(f"Retained Customers (Active 60d)  : {n_total - n_churn:,} ({1 - churn_rate:.2%})")

    # 2. Stratified train/test split (75/25)
    X = dataset[FEATURES]
    y = dataset["churn"]
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        dataset["CustomerID"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    class_counts = y_train.value_counts()
    if 0 not in class_counts or 1 not in class_counts:
        raise ValueError("Training split must contain both churn classes.")
    positive_weight = class_counts[0] / class_counts[1]

    print(f"\nTrain Set: {len(X_train):,} customers | Test Holdout: {len(X_test):,} customers")
    print(f"Class Balance Weight: {positive_weight:.3f}")

    # 3. Fit XGBoost Classifier
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=350,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save model
    model.save_model(str(MODEL_JSON))
    print(f"Model saved to: {MODEL_JSON}")

    # 4. Predict on Test Set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    # 5. Evaluate Metrics
    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    prec_top_20 = precision_at_top_fraction(y_test, test_probs, fraction=0.20)
    test_churn_rate = float(y_test.mean())

    print("\n" + "=" * 70)
    print(" Churn Model Evaluation on Held-Out Test Set (25% Split)")
    print("=" * 70)
    print(f"  ROC-AUC Score          : {roc_auc:.4f}")
    print(f"  Average Precision (PR) : {pr_auc:.4f}")
    print(f"  Precision @ Top 20%    : {prec_top_20:.4f}  (Baseline Churn Rate: {test_churn_rate:.4f})")
    print(f"  Lift @ Top 20%         : {prec_top_20 / test_churn_rate:.2f}x random targeting")
    print(f"  Brier Score Loss       : {brier:.4f}")
    print("-" * 70)
    print("\nClassification Report (Threshold = 0.5):")
    print(classification_report(y_test, test_preds, digits=4))

    # 6. Save Predictions CSV
    predictions_df = pd.DataFrame({
        "CustomerID": id_test.values,
        "actual_churn": y_test.values,
        "predicted_churn": test_preds,
        "churn_probability": test_probs.round(4),
    }).sort_values("churn_probability", ascending=False)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved holdout predictions: {PREDICTIONS_CSV}")

    # 7. SHAP Explanations
    print("\nComputing SHAP feature importance...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    print(f"Saved ranked SHAP importance: {SHAP_IMPORTANCE_CSV}")

    print("\nTop 7 Churn Drivers (SHAP Mean Absolute Value):")
    for rank, row in importance_df.head(7).iterrows():
        print(f"  {rank + 1}. {row['feature']:<22}: {row['mean_abs_shap']:.4f}")

    # Generate visual SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = importance_df.head(10).iloc[::-1]
    ax.barh(top_n["feature"], top_n["mean_abs_shap"], color="#3498DB", edgecolor="white", height=0.6)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)", fontsize=10)
    ax.set_title("Customer Churn Feature Importance (SHAP)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(SHAP_SUMMARY_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved SHAP importance chart: {SHAP_SUMMARY_PNG}")

    # 8. Save Metrics JSON
    metrics_payload = {
        "observation_cutoff": str(CUTOFF_DATE.date()),
        "label_window_days": LABEL_WINDOW_DAYS,
        "label_end_date": str(LABEL_END_DATE.date()),
        "total_active_customers": n_total,
        "total_churned_customers": int(n_churn),
        "overall_churn_rate": round(churn_rate, 4),
        "test_holdout_size": len(X_test),
        "test_churn_rate": round(test_churn_rate, 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision_pr_auc": round(pr_auc, 4),
        "precision_at_top_20_pct": round(prec_top_20, 4),
        "lift_at_top_20_pct": round(prec_top_20 / test_churn_rate, 2),
        "brier_score_loss": round(brier, 4),
        "top_features": importance_df["feature"].head(5).tolist(),
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics payload: {METRICS_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
