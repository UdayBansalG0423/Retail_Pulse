"""
evaluate_clusters.py
--------------------
Compares K-Means cluster counts k=2..7 using inertia and silhouette score,
then saves an elbow chart and a silhouette chart to reports/segmentation/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on all platforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "customer_rfm.csv"
OUTPUT_DIR = ROOT / "reports" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rfm = pd.read_csv(DATA_FILE)

    features = ["recency", "frequency", "monetary"]
    X_log = np.log1p(rfm[features])
    X_scaled = StandardScaler().fit_transform(X_log)

    results = []
    for k in range(2, 8):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "inertia": round(model.inertia_, 2),
            "silhouette_score": round(sil, 4),
        })
        print(f"  k={k}  inertia={model.inertia_:,.1f}  silhouette={sil:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "cluster_evaluation.csv", index=False)

    print("\n--- CLUSTER EVALUATION TABLE ---")
    print(results_df.to_string(index=False))

    # ── Elbow chart ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("K-Means Cluster Evaluation", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(results_df["k"], results_df["inertia"],
             marker="o", linewidth=2, color="#2563EB")
    ax1.set_xlabel("Number of clusters (k)", fontsize=11)
    ax1.set_ylabel("Inertia (within-cluster sum of squares)", fontsize=11)
    ax1.set_title("Elbow Chart — look for the bend", fontsize=12)
    ax1.set_xticks(results_df["k"])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── Silhouette chart ──────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(results_df["k"], results_df["silhouette_score"],
             marker="o", linewidth=2, color="#16A34A")
    ax2.set_xlabel("Number of clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score (higher = better separation)", fontsize=11)
    ax2.set_title("Silhouette Score by k", fontsize=12)
    ax2.set_xticks(results_df["k"])
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "cluster_evaluation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nChart saved: {out_path}")


if __name__ == "__main__":
    main()
