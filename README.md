# RetailPulse — AI-Powered Retail Analytics Platform

**RetailPulse** is an enterprise-grade retail analytics platform combining customer behavioral clustering, demand forecasting, and inventory replenishment optimization into an interactive, multi-page Streamlit decision cockpit.

---

## 🚀 Key Modules

### 1. 👥 Customer Segmentation (RFM + K-Means)
- Analyzes over **524,000+** validated transactions across **4,338** unique customers.
- Extracts Recency, Frequency (distinct purchase dates proxy), and Monetary features with robust log-transformation and standard scaling.
- Discovers 4 distinct behavioral clusters:
  - **Champions**: High monetary value, frequent recent purchases.
  - **Loyal Customers**: Consistent repeat shoppers with moderate recency.
  - **At-Risk Customers**: High past value who have not purchased recently.
  - **Inactive / Hibernating**: Low frequency and long dormancy.
- Provides dynamic filtering by segment, spend, and recency, plus CSV customer export.

### 2. 📈 Demand Forecasting (Seasonal-Naïve Baseline vs. Meta Prophet)
- Aggregates daily transaction volume into contiguous time series with strict zero-filling for closed/no-sales days.
- **Benchmark Comparison (42-Day Holdout Window: Oct 28 – Dec 08, 2011)** on pilot SKU (`WHITE HANGING HEART T-LIGHT HOLDER`):
  - **Seasonal-Naïve Baseline (Lag-7):** MAE = `117.95` units/day | WAPE = `81.32%`
  - **Meta Prophet:** MAE = `108.44` units/day | WAPE = `74.76%` (**8.1% MAE improvement**)
- Generates 30-day forward demand projections refitted on full historical data.

### 3. 📦 Inventory Optimization & Replenishment
- Demand-driven replenishment engine based on target stock $(s, S)$ inventory theory:
  - **Inventory Position:** $IP = \text{Current Stock} + \text{On Order} - \text{Backorders}$
  - **Reorder Point:** $ROP = (d \times L) + SS$
  - **Target Stock:** $T = d \times (L + R) + SS$
  - **Suggested Order Quantity:** $Q = \max(0, \lceil T - IP \rceil)$
- Action classification: `REORDER_NOW` (High urgency), `MONITOR` (Medium urgency), and `NO_REORDER` (Low urgency).
- Full auditability with automated unit test suite.
- Explicit `SIMULATED_DEMO_DATA` labeling distinguishing real demand signals from simulated warehouse parameters.

---

## 📁 Repository Structure

```
RetailPulse/
├── dashboard/
│   ├── app.py                          # Executive Hub & KPI cockpit
│   └── pages/
│       ├── 1_Customer_Segmentation.py   # RFM clustering & Customer Explorer
│       ├── 2_Demand_Forecasting.py      # Prophet evaluation & 30-day curves
│       └── 3_Inventory_Optimization.py  # Stock coverage, ROP & orders
├── src/
│   ├── data/                           # Ingestion, validation & cleaning
│   ├── segmentation/                   # RFM modeling & cluster evaluation
│   ├── forecasting/                    # Data aggregation, baseline & Prophet
│   └── inventory/                      # Sample inventory & optimizer engine
├── tests/
│   └── test_inventory_optimizer.py     # Unit tests for replenishment rules
├── reports/                            # Generated evaluation charts
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🛠️ Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/UdayBansalG0423/Retail_Pulse.git
cd Retail_Pulse
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Pipelines
```bash
# Data preparation & segmentation
python src/segmentation/rfm_segmentation.py
python src/segmentation/label_and_visualize.py

# Forecasting pipeline
python src/forecasting/prepare_sales.py
python src/forecasting/forecast_baseline.py
python src/forecasting/prophet_forecast.py
python src/forecasting/generate_future_forecast.py

# Inventory recommendation engine
python src/inventory/create_sample_inventory.py
python src/inventory/inventory_optimizer.py
```

### 3. Run Automated Tests
```bash
python -m unittest tests/test_inventory_optimizer.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Visit `http://localhost:8501` to view the platform.

---

## 📜 License & Acknowledgements
- Dataset: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from UCI Machine Learning Repository.
- Internship Project for Zidio Development.
