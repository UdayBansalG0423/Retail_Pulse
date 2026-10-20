"""
inventory_optimizer.py
----------------------
Calculates inventory replenishment recommendations by combining 30-day demand
forecasts from Meta Prophet with inventory parameters (current stock, on-order,
backorders, supplier lead time, safety stock, and review period).

Methodology & Formulas
----------------------
1. Average Daily Demand:
   d = sum(yhat_t for t=1..30) / 30

2. Inventory Position:
   IP = current_stock + on_order - backorders

3. Reorder Point (ROP):
   ROP = (d * lead_time_days) + safety_stock

4. Target Stock (T):
   T = d * (lead_time_days + review_period_days) + safety_stock

5. Suggested Order Quantity (Q):
   Q = max(0, ceil(T - IP))

6. Days of Stock Coverage:
   days_of_stock = current_stock / d (if d > 0 else 999)

7. Recommended Action & Urgency:
   - REORDER_NOW: IP <= ROP (Urgency: HIGH)
   - MONITOR    : ROP < IP <= ROP * 1.2 (Urgency: MEDIUM)
   - ADEQUATE   : IP > ROP * 1.2 (Urgency: LOW)

Inputs:
  - data/processed/forecast_30_day.csv
  - data/processed/sample_inventory.csv (DEMO / SIMULATED)

Output:
  - data/processed/inventory_recommendations.csv
"""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORECAST_FILE = ROOT / "data" / "processed" / "forecast_30_day.csv"
INVENTORY_FILE = ROOT / "data" / "processed" / "sample_inventory.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "inventory_recommendations.csv"

REQUIRED_FORECAST = {"product", "date", "yhat"}
REQUIRED_INVENTORY = {
    "product",
    "current_stock",
    "on_order",
    "backorders",
    "lead_time_days",
    "safety_stock",
    "review_period_days",
}


def load_inputs():
    """Validate and load forecast and inventory input CSVs."""
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FORECAST_FILE}. Generate the 30-day future forecast first."
        )
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INVENTORY_FILE}. Run create_sample_inventory.py first."
        )

    forecast_df = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
    inventory_df = pd.read_csv(INVENTORY_FILE)

    missing_f = REQUIRED_FORECAST - set(forecast_df.columns)
    if missing_f:
        raise ValueError(f"Forecast file missing required columns: {missing_f}")

    missing_i = REQUIRED_INVENTORY - set(inventory_df.columns)
    if missing_i:
        raise ValueError(f"Inventory file missing required columns: {missing_i}")

    return forecast_df, inventory_df


def calculate_inventory_metrics(
    current_stock: float,
    on_order: float,
    backorders: float,
    lead_time_days: float,
    safety_stock: float,
    review_period_days: float,
    avg_daily_demand: float,
) -> dict:
    """Core mathematical inventory calculations with input validation."""
    if lead_time_days < 0:
        raise ValueError(f"Lead time cannot be negative: {lead_time_days}")
    if safety_stock < 0:
        raise ValueError(f"Safety stock cannot be negative: {safety_stock}")
    if review_period_days <= 0:
        raise ValueError(f"Review period must be positive: {review_period_days}")
    if avg_daily_demand < 0:
        raise ValueError(f"Demand cannot be negative: {avg_daily_demand}")

    # 1. Inventory Position
    ip = current_stock + on_order - backorders

    # Explicit handling for zero-demand forecasts
    if avg_daily_demand == 0.0:
        rop = safety_stock
        target_stock = safety_stock
        suggested_order = max(0, math.ceil(target_stock - ip))
        days_of_stock = 999.0
        if ip < safety_stock:
            action = "REORDER_NOW"
            urgency = "HIGH"
            rationale = (
                f"Zero daily demand forecasted. Inventory position ({ip:.0f}) is below "
                f"safety stock ({safety_stock:.0f}). Order {suggested_order} units to replenish safety buffer."
            )
        else:
            action = "NO_REORDER"
            urgency = "LOW"
            rationale = (
                f"Zero daily demand forecasted and inventory position ({ip:.0f}) meets safety stock ({safety_stock:.0f}). "
                "No reorder required."
            )
        return {
            "inventory_position": ip,
            "reorder_point": rop,
            "target_stock": target_stock,
            "suggested_order_qty": suggested_order,
            "days_of_stock": days_of_stock,
            "action": action,
            "urgency": urgency,
            "rationale": rationale,
        }

    # 2. Reorder Point (ROP)
    rop = (avg_daily_demand * lead_time_days) + safety_stock

    # 3. Target Stock (T)
    target_stock = avg_daily_demand * (lead_time_days + review_period_days) + safety_stock

    # 4. Suggested Order Quantity (Q)
    suggested_order = max(0, math.ceil(target_stock - ip))

    # 5. Days of Stock Coverage (physical on-hand stock)
    days_of_stock = round(current_stock / avg_daily_demand, 1)

    # 6. Action Rule & Priority
    if ip <= rop:
        action = "REORDER_NOW"
        urgency = "HIGH"
        rationale = (
            f"Inventory position ({ip:.0f}) is at or below ROP ({rop:.1f}). "
            f"Order {suggested_order} units immediately to avoid stockout during lead time."
        )
    elif ip <= rop * 1.2:
        action = "MONITOR"
        urgency = "MEDIUM"
        rationale = (
            f"Inventory position ({ip:.0f}) is within 20% of ROP ({rop:.1f}). "
            f"Prepare replenishment order for next review cycle."
        )
    else:
        action = "NO_REORDER"
        urgency = "LOW"
        rationale = (
            f"Stock levels sufficient ({days_of_stock} days coverage). "
            f"Inventory position ({ip:.0f}) comfortably exceeds ROP ({rop:.1f}). No order needed."
        )

    return {
        "inventory_position": ip,
        "reorder_point": rop,
        "target_stock": target_stock,
        "suggested_order_qty": suggested_order,
        "days_of_stock": days_of_stock,
        "action": action,
        "urgency": urgency,
        "rationale": rationale,
    }


def compute_recommendations(forecast_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Compute replenishment metrics and recommendations for each product."""
    recommendations = []

    # Calculate 30-day forecast demand metrics per product
    demand_summary = (
        forecast_df.groupby("product")
        .agg(
            total_forecast_demand=("yhat", "sum"),
            forecast_days=("date", "count"),
            avg_daily_demand=("yhat", "mean"),
        )
        .reset_index()
    )

    merged = inventory_df.merge(demand_summary, on="product", how="inner")

    if merged.empty:
        raise ValueError("No matching products found between forecast and inventory files!")

    for _, row in merged.iterrows():
        prod = row["product"]
        f_days = int(row["forecast_days"])
        if f_days != 30:
            raise ValueError(
                f"Product '{prod}' has {f_days} forecast days, expected exactly 30 days."
            )
        metrics = calculate_inventory_metrics(
            current_stock=float(row["current_stock"]),
            on_order=float(row["on_order"]),
            backorders=float(row["backorders"]),
            lead_time_days=float(row["lead_time_days"]),
            safety_stock=float(row["safety_stock"]),
            review_period_days=float(row["review_period_days"]),
            avg_daily_demand=float(row["avg_daily_demand"]),
        )
        data_source = row.get("data_type", "SIMULATED_DEMO_DATA")

        recommendations.append({
            "product": prod,
            "current_stock": int(row["current_stock"]),
            "on_order": int(row["on_order"]),
            "backorders": int(row["backorders"]),
            "inventory_position": int(metrics["inventory_position"]),
            "lead_time_days": int(row["lead_time_days"]),
            "safety_stock": int(row["safety_stock"]),
            "review_period_days": int(row["review_period_days"]),
            "forecast_days": int(row["forecast_days"]),
            "avg_daily_demand": round(float(row["avg_daily_demand"]), 2),
            "days_of_stock": metrics["days_of_stock"],
            "reorder_point": round(metrics["reorder_point"], 1),
            "target_stock": round(metrics["target_stock"], 1),
            "suggested_order_qty": int(metrics["suggested_order_qty"]),
            "action": metrics["action"],
            "urgency": metrics["urgency"],
            "rationale": metrics["rationale"],
            "data_source": data_source,
        })

    return pd.DataFrame(recommendations)


def main():
    print("=" * 70)
    print(" RetailPulse — Inventory Recommendation & Optimization Engine")
    print("=" * 70)

    forecast_df, inventory_df = load_inputs()
    print(f"Loaded {len(forecast_df)} forecast entries across {forecast_df['product'].nunique()} products.")
    print(f"Loaded {len(inventory_df)} inventory records.")

    rec_df = compute_recommendations(forecast_df, inventory_df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nRecommendations successfully saved to: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print(f" {'Product':<32} | {'Action':<11} | {'IP':<6} | {'ROP':<7} | {'Order Qty':<9}")
    print("-" * 70)
    for _, r in rec_df.iterrows():
        p_name = r["product"][:30]
        print(f" {p_name:<32} | {r['action']:<11} | {r['inventory_position']:<6} | {r['reorder_point']:<7.1f} | {r['suggested_order_qty']:<9}")
    print("=" * 70)

    print("\nDetailed Recommendations:")
    for _, r in rec_df.iterrows():
        print(f"\n* Product: {r['product']}")
        print(f"  - Action          : [{r['action']}] (Urgency: {r['urgency']})")
        print(f"  - Avg Daily Demand: {r['avg_daily_demand']} units/day")
        print(f"  - Stock Coverage  : {r['days_of_stock']} days on hand")
        print(f"  - Inv Position    : {r['inventory_position']} (Stock: {r['current_stock']} + OnOrder: {r['on_order']} - Backorders: {r['backorders']})")
        print(f"  - Reorder Point   : {r['reorder_point']}")
        print(f"  - Target Stock    : {r['target_stock']}")
        print(f"  - Suggested Order : {r['suggested_order_qty']} units")
        print(f"  - Rationale       : {r['rationale']}")
        print(f"  - Notice          : {r['data_source']} (Illustrative stock/lead times)")


if __name__ == "__main__":
    main()
