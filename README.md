# RetailPulse — AI-Powered Retail Analytics Platform

**RetailPulse** is an end-to-end retail analytics platform combining customer behavioural segmentation, demand forecasting, inventory replenishment optimisation, and customer churn prediction into an interactive multi-page Streamlit dashboard.

> **Internship project — Zidio Development.**  
> Built on the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (December 2010 – December 2011, UK-based transactions).

---

## 🚀 Module Summary

| # | Module | Method | Key Measured Result |
|---|--------|--------|-------------------|
| 1 | Customer Segmentation | RFM + K-Means (k=4) | 4,338 customers → 4 segments |
| 2 | Demand Forecasting | Seasonal-Naïve vs Meta Prophet | Prophet WAPE 74.76% vs Baseline 81.32% on 42-day holdout |
| 3 | Inventory Optimisation | Target-Stock (s, S) policy | Simulated inputs — see notice below |
| 4 | Churn Prediction | XGBoost + SHAP | ROC-AUC 0.7422 · Precision@Top-20% 75.14% on held-out test cohort |

> ⚠️ **Accuracy targets not yet demonstrated.**  
> The assignment specifies MAPE ≤ 12% for forecasting and ROC-AUC ≥ 0.88 for churn.  
> Measured results on held-out data are 74.76% WAPE and 0.7422 AUC respectively.  
> These are reported honestly; neither target has been achieved on this dataset.

---

## 📋 Module Details

### 1. 👥 Customer Segmentation (RFM + K-Means)

- **Data:** 524,000+ validated transaction rows from the UCI Online Retail dataset.
- **Frequency definition:** Distinct purchase dates per customer (not raw invoice count), to avoid inflating repeat-line items in a single basket.
- **Pipeline:** Log-transformation → standard scaling → K-Means (k=4, 10 random initialisations).
- **Segments:**
  - **Champions** — High spend, frequent recent activity.
  - **Loyal Customers** — Consistent repeat shoppers with moderate recency.
  - **At-Risk Customers** — High historical value with long purchase gaps.
  - **Inactive / Hibernating** — Low frequency and extended dormancy.
- **Dashboard features:** Real-time segment, spend, and recency filters; customer ID lookup; CSV export.

### 2. 📈 Demand Forecasting (Seasonal-Naïve Baseline vs. Meta Prophet)

- **Pilot SKU:** `WHITE HANGING HEART T-LIGHT HOLDER`
- **Holdout window:** 42 days — 2011-10-28 to 2011-12-08.
- **Zero-filling:** Closed/no-sale days explicitly set to zero before modelling.
- **Measured results (held-out test set only):**

  | Model | MAE (units/day) | WAPE |
  |-------|----------------|------|
  | Seasonal-Naïve Baseline (Lag-7) | 117.95 | 81.32% |
  | Meta Prophet | **108.44** | **74.76%** |

- **Forward projections:** 30-day demand forecast generated from the full historical series.
- **Note:** The assignment MAPE ≤ 12% target has **not** been demonstrated. The dataset contains extreme demand spikes and many zero-sale days that inflate percentage errors.

### 3. 📦 Inventory Optimisation & Replenishment

- **Policy:** Continuous-review Target-Stock $(s, S)$ system.
- **Formulas:**
  - $IP = \text{Stock} + \text{On Order} - \text{Backorders}$
  - $ROP = (d \times L) + SS$
  - $T = d \times (L + R) + SS$
  - $Q = \max(0, \lceil T - IP \rceil)$
- **⚠️ Simulated inputs:** Current stock levels, supplier lead times, review periods, and backorders are explicitly labelled `SIMULATED_DEMO_DATA`. The demand signal $d$ comes from the real 30-day Prophet forecast; warehouse parameters are invented for algorithmic validation only.
- **Actions:** `REORDER_NOW` (urgent), `MONITOR` (watch), `NO_REORDER` (sufficient stock).
- **What-If Simulator** on the dashboard allows interactive lead-time and stock-level scenario testing.

### 4. 🔮 Churn Prediction (XGBoost + SHAP)

- **Churn definition:** Customer has **no transaction** in the 60-day window following the observation cutoff (2011-09-30 → 2011-11-29).
- **Features:** Recency, frequency (distinct dates), total spend, average order value, unique products, spend trend, tenure, and recent 30-day spend.
- **Evaluation (25% stratified held-out test set):**

  | Metric | Value |
  |--------|-------|
  | ROC-AUC | 0.7422 |
  | PR-AUC | 0.7149 |
  | Precision@Top-20% | 75.14% |
  | Lift@Top-20% | 1.42× |
  | Brier Score | 0.2034 |

- **Top predictors (mean \|SHAP\|):** `frequency_dates` (0.4275) → `unique_products` (0.3046) → `recency_days` (0.2312).
- **⚠️ Scope limitation:** The churn page scores a **held-out historical test cohort**, not current live customers. Results should not be interpreted as production-ready campaign recommendations. The assignment target of ROC-AUC ≥ 0.88 has **not** been demonstrated.

---

## 📁 Repository Structure

```
RetailPulse/
├── .github/workflows/
│   ├── ci.yml                           # Core regression test CI
│   └── tests.yml                        # Full dependency test CI
├── dashboard/
│   ├── app.py                           # Executive Hub & 5-metric KPI cockpit
│   └── pages/
│       ├── 1_Customer_Segmentation.py   # RFM clustering & Customer Explorer
│       ├── 2_Demand_Forecasting.py      # Prophet evaluation & 30-day curves
│       ├── 3_Inventory_Optimization.py  # Stock coverage, ROP & What-If Simulator
│       └── 4_Churn_Prediction.py        # XGBoost risk tiers, SHAP, customer lookup
├── src/
│   ├── data/                            # Ingestion, validation & cleaning
│   ├── segmentation/                    # RFM modelling & cluster evaluation
│   ├── forecasting/                     # Data aggregation, baseline & Prophet
│   ├── inventory/                       # Sample inventory & optimiser engine
│   └── churn/                           # XGBoost training & SHAP interpretability
├── tests/
│   ├── test_customer_segmentation.py    # Segmentation filter regression tests
│   ├── test_forecasting.py              # Forecast holdout alignment tests
│   ├── test_inventory_optimizer.py      # Replenishment rule unit tests (8 rules)
│   └── test_churn_prediction.py         # Churn metric & probability bound tests
├── models/
│   └── churn_xgb.json                   # Trained XGBoost model (tracked in Git)
├── reports/
│   └── churn/
│       ├── churn_metrics.json           # Held-out evaluation metrics
│       ├── churn_shap_importance.csv    # Mean |SHAP| per feature
│       └── churn_shap_summary.png       # SHAP beeswarm visualisation
├── data/
│   └── processed/                       # Generated CSVs (git-ignored; see setup below)
├── requirements.txt
└── README.md
```

---

## 🛠️ Quickstart — Fresh Clone Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/UdayBansalG0423/Retail_Pulse.git
cd Retail_Pulse

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the full pipeline (generates all ignored processed data)

Run each script in the order shown. Scripts that depend on earlier outputs will fail if you skip steps.

```bash
# ── Step 1: Data preparation & segmentation ───────────────────────────────
python src/data/ingest_data.py          # Downloads & validates UCI dataset
python src/segmentation/rfm_segmentation.py
python src/segmentation/label_and_visualize.py
# Output: data/processed/customer_rfm.csv, data/processed/segment_summary.csv

# ── Step 2: Demand forecasting ────────────────────────────────────────────
python src/forecasting/prepare_sales.py
python src/forecasting/forecast_baseline.py
python src/forecasting/prophet_forecast.py
python src/forecasting/generate_future_forecast.py
# Output: data/processed/forecast_30_day.csv, data/processed/*_test_predictions.csv

# ── Step 3: Inventory optimisation ───────────────────────────────────────
python src/inventory/create_sample_inventory.py
python src/inventory/inventory_optimizer.py
# Output: data/processed/inventory_recommendations.csv

# ── Step 4: Churn prediction ──────────────────────────────────────────────
python src/churn/churn_prediction.py
# Output: data/processed/churn_customer_predictions.csv
#         reports/churn/churn_metrics.json
#         reports/churn/churn_shap_*.{csv,png}
#         models/churn_xgb.json
```

### 3. Run the automated test suite

```bash
python -m compileall -q src dashboard tests
python -m unittest discover -s tests -v
```

Expected: **19 tests, 0 failures** (3 churn tests skip when `xgboost` is not installed).

### 4. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser. All four module pages are available in the left sidebar.

---

## 🔄 Reproducibility & Data Notes

| Concern | Detail |
|---------|--------|
| **Raw data** | Excluded from Git (`data/raw/*`). Re-downloaded by `src/data/ingest_data.py` via `ucimlrepo`. |
| **Processed CSVs** | Excluded from Git (`data/processed/*`). Regenerated by the pipeline scripts above. |
| **Churn model** | `models/churn_xgb.json` is tracked in Git (415 KB). The XGBoost `RANDOM_STATE=42` seed makes training deterministic. |
| **SHAP reports** | `reports/churn/` is tracked in Git. |
| **Simulated inventory** | Stock levels, lead times, and backorders are invented values labelled `SIMULATED_DEMO_DATA`. |
| **Dataset** | UCI Online Retail (not Online Retail II). Date range: 2010-12-01 to 2011-12-09. UK customers only after cleaning. |

---

## ✅ CI / Automated Tests

GitHub Actions runs two workflows on every push to `main`:

- **`ci.yml`** — Fast core tests (no `prophet`/`xgboost` required). Runs in ~30 s.
- **`tests.yml`** — Full dependency install including optional ML packages.

---

## 📜 License & Acknowledgements

- **Dataset:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) — Dr. Daqing Chen, London South Bank University. UCI Machine Learning Repository, 2015.
- **Internship project** for Zidio Development.
