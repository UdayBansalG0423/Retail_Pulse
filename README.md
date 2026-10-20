<div align="center">

# 🛍️ RetailPulse
### AI-Powered Retail Analytics Platform

[![CI](https://github.com/UdayBansalG0423/Retail_Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/UdayBansalG0423/Retail_Pulse/actions/workflows/ci.yml)
[![Tests](https://github.com/UdayBansalG0423/Retail_Pulse/actions/workflows/tests.yml/badge.svg)](https://github.com/UdayBansalG0423/Retail_Pulse/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Churn_Model-orange)
![Prophet](https://img.shields.io/badge/Meta_Prophet-Forecasting-blueviolet)

**Enterprise retail decision intelligence: Sales Analytics · Customer RFM & Churn · Demand Forecasting · Inventory Replenishment**

*Internship project — Zidio Development · Dataset: UCI Online Retail (Dec 2010 – Dec 2011)*

</div>

---

## 📸 Dashboard Overview & Multi-Page Navigation

RetailPulse delivers an executive cockpit with four dedicated, properly structured analytical dashboards in Streamlit:

![RetailPulse Executive Dashboard](docs/assets/executive_overview.png)
*Executive Hub — Cross-module KPI Cockpit with direct links to all four analytical pages*

---

## 🧭 Dashboard Architecture & Workflow

RetailPulse transforms raw transactional data into four interconnected operational dashboards in proper order:

| Page | Dashboard Name | Core Capabilities | Key Technologies / Metrics |
|:---:|---|---|---|
| **1** | **💰 Sales Dashboard** | Revenue trends, products, sales volume, and country-level analytics | 524K+ transactions · £8.9M revenue · 38 export markets |
| **2** | **👥 Customer Dashboard** | RFM segmentation, customer explorer, and XGBoost churn prediction | 4,338 customers · ROC-AUC 0.7422 · SHAP explainability |
| **3** | **📈 Forecast Dashboard** | Prophet vs baseline evaluation, holdout charts, and 30-day predictions | 42-day holdout · WAPE 74.76% · 30-day forward demand curves |
| **4** | **📦 Inventory Dashboard** | Reorder recommendations, stock coverage, and interactive what-if analysis | Continuous $(s, S)$ Target Stock policy · Stock vs ROP · Lead time stress-testing |

> ⚠️ **Honest metric reporting:** The assignment specification mentions MAPE ≤ 12% and ROC-AUC ≥ 0.88. Measured results on held-out test data are **WAPE 74.76%** (improving on 81.36% baseline) and **ROC-AUC 0.7422** (1.42× lift @ top-20%). Neither theoretical target has been claimed without rigorous holdout evidence.

---

## 📊 Detailed Page Walkthroughs

### Page 1 — 💰 Sales Dashboard
**Revenue trends, products, sales volume, and country-level analytics.**

![Sales Dashboard](docs/assets/sales_dashboard.png)

- **Revenue Trends & Seasonality:** Interactive toggle between Daily, Weekly, and Monthly aggregations with 7-day moving averages and day-of-week shopping volume patterns.
- **Top Product Performance & Velocity:** Horizontal bar charts displaying Top N revenue drivers and sales volume leaders alongside a searchable catalog with margin shares.
- **Geographic Market Distribution:** UK domestic vs. 37 International export destinations with market concentration pies and country breakdown tables.
- **Transaction Explorer:** Filtered tabular records with date pickers, country multiselect, product substring search, and CSV export.

---

### Page 2 — 👥 Customer Dashboard
**RFM segmentation and churn prediction, including model metrics and SHAP explanations.**

![Customer Dashboard](docs/assets/customer_dashboard.png)

- **Behavioral RFM Clustering:** K-Means clustering ($k=4$) segmenting 4,338 accounts into **Champions**, **Loyal Customers**, **At-Risk**, and **Inactive** cohorts.
- **Customer Explorer:** Real-time multi-slider filtering on Recency, Frequency, and Spend bounds with instant CSV export.
- **XGBoost Churn Prediction:** Classifies disengagement risk over a 60-day observation window (`2011-10-01` → `2011-11-29`). Achieves **0.7422 ROC-AUC**, **75.14% Precision@Top-20%**, and **1.42× targeting lift**.
- **SHAP Interpretability:** Feature attribution showing why frequency, diversity, and recency drive customer attrition:

| Rank | Feature | Mean \|SHAP\| | Business Interpretation |
|:---:|---|:---:|---|
| 1 | `frequency_dates` | 0.4275 | Customers with fewer distinct order dates churn at significantly higher rates |
| 2 | `unique_products` | 0.3046 | Broader product diversity indicates relationship stickiness |
| 3 | `recency_days` | 0.2312 | Long purchase dormancy strongly indicates account attrition |
| 4 | `total_units` | 0.1707 | Higher purchase volume correlates with ongoing account retention |
| 5 | `avg_order_value` | 0.1400 | Higher basket spend signals committed retail accounts |

- **Customer 360° Lookup:** Search any Customer ID to inspect their cluster, lifetime spend, churn probability gauge, and prescriptive retention action.

---

### Page 3 — 📈 Forecast Dashboard
**Prophet versus baseline evaluation, forecast charts, and 30-day predictions.**

![Forecast Dashboard](docs/assets/forecast_dashboard.png)

- **Benchmark Evaluation:** Compares Meta Prophet against a Seasonal-Naïve benchmark (Lag-7) across a common 42-day held-out test window:

| Model | MAE (units/day) | WAPE | Model Characteristics |
|---|:---:|:---:|---|
| Seasonal-Naïve Baseline | 158.41 | 81.36% | Weekly seasonal lag; echoes anomalies |
| **Meta Prophet** | **108.44** | **74.76%** | **−49.97 u/day improvement (−6.60 pp WAPE)** |

- **Visual Evaluation Chart:** Clear overlay of actual vs. predicted units illustrating Prophet's outlier smoothing.
- **30-Day Forward Projections:** Interactive product selector with daily projected unit demand curves, total volume sums, and CSV download for procurement schedules.
- **Diagnostics:** Analysis of zero-demand days and large residual outliers.

---

### Page 4 — 📦 Inventory Dashboard
**Reorder recommendations, stock coverage, and interactive what-if analysis.**

![Inventory Dashboard](docs/assets/inventory_dashboard.png)

- **Target Stock Replenishment Policy:** Operates a continuous review $(s, S)$ policy powered by Prophet's 30-day forward demand rates:
  $$\text{Inventory Position (IP)} = \text{Current Stock} + \text{On Order} - \text{Backorders}$$
  $$\text{Reorder Point (ROP)} = (d \times L) + SS$$
  $$\text{Target Stock (T)} = d \times (L + R) + SS$$
  $$\text{Order Quantity (Q)} = \max(0, \lceil T - IP \rceil)$$
- **Urgency Alert Matrix:** Flags SKUs as `REORDER_NOW`, `MONITOR`, or `NO_REORDER` with days-of-stock runout estimates.
- **Stock Coverage vs. Lead Time:** Dual bar charts contrasting physical days on hand against supplier delivery lead times.
- **Interactive What-If Simulator:** Dynamic sliders for lead-time delays (1–30 days), safety stock multipliers (0.2×–3.0×), and demand surge surges (0.5×–3.0×) with real-time recalculation of suggested orders and decision actions.

---

## 🏗️ Repository Structure

```
RetailPulse/
├── .github/workflows/
│   ├── ci.yml                          # Fast regression tests (~30s)
│   └── tests.yml                       # Full dependency test suite
│
├── dashboard/
│   ├── app.py                          # Executive Hub & navigation entrypoint
│   └── pages/
│       ├── 1_Sales_Dashboard.py        # Page 1: Revenue, products, volume & countries
│       ├── 2_Customer_Dashboard.py     # Page 2: RFM segmentation & Churn with SHAP
│       ├── 3_Forecast_Dashboard.py     # Page 3: Prophet vs baseline & 30-day forecast
│       └── 4_Inventory_Dashboard.py    # Page 4: Reorders, stock coverage & What-If
│
├── src/
│   ├── data/                           # Data ingestion & cleaning
│   ├── segmentation/                   # RFM clustering & evaluation
│   ├── forecasting/                    # Baseline, Prophet & 30-day projections
│   ├── inventory/                      # Replenishment engine & What-If calculations
│   └── churn/                          # XGBoost training & SHAP interpretability
│
├── tests/
│   ├── test_customer_segmentation.py   # Customer filter regression tests
│   ├── test_forecasting.py             # Forecast alignment & non-negativity tests
│   ├── test_inventory_optimizer.py     # Replenishment policy rule tests
│   └── test_churn_prediction.py        # Model metrics & bound tests
│
├── models/
│   └── churn_xgb.json                  # Tracked XGBoost model artifact (~415 KB)
│
├── reports/
│   ├── churn/                          # Churn metrics, SHAP importance & plots
│   ├── forecasting/                    # Holdout forecast visual evaluation plots
│   └── segmentation/                   # Segment distribution & RFM profile plots
│
├── docs/assets/                        # Visual screenshots and artifacts
├── setup_pipeline.py                   # Automated end-to-end pipeline runner
└── requirements.txt
```

---

## ⚡ Quickstart

### 1 · Clone and set up environment

```bash
git clone https://github.com/UdayBansalG0423/Retail_Pulse.git
cd Retail_Pulse

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 2 · Run the automated pipeline

```bash
python setup_pipeline.py
```

This runs all 5 stages in order and verifies outputs:
1. Stage 1 · Data ingestion & cleaning
2. Stage 2 · Customer segmentation
3. Stage 3 · Demand forecasting
4. Stage 4 · Inventory optimisation
5. Stage 5 · Churn prediction

### 3 · Run regression tests

```bash
python -m compileall -q src dashboard tests
python -m unittest discover -s tests -v
```

Expected output:
```
Ran 19 tests in ~0.03s
OK (skipped=3)
```

### 4 · Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser. All four pages appear in the sidebar in proper order:
- **Page 1: Sales Dashboard**
- **Page 2: Customer Dashboard**
- **Page 3: Forecast Dashboard**
- **Page 4: Inventory Dashboard**

---

## 🧪 Test Coverage & CI

| Test File | Tests | Coverage Scope | Status |
|---|:---:|---|:---:|
| `test_customer_segmentation.py` | 4 | Slider bounds, maximum spend integrity, segment summation | ✅ Passing |
| `test_forecasting.py` | 4 | Date alignment, actuals matching, MAE improvement, non-negativity | ✅ Passing |
| `test_inventory_optimizer.py` | 8 | Target stock formula, ROP logic, backorders, lead time guards | ✅ Passing |
| `test_churn_prediction.py` | 3 | ROC-AUC, PR-AUC, precision@top-20%, probability bounds | ✅ Passing |
| **Total** | **19** | **Full end-to-end regression validation** | **100% CI Green** |

---

## 📜 License & Acknowledgements

- **Dataset:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) — Dr. Daqing Chen, London South Bank University.
- **Internship Project:** Developed for [Zidio Development](https://zidio.in).
- **Core Stack:** [Streamlit](https://streamlit.io) · [Meta Prophet](https://facebook.github.io/prophet/) · [XGBoost](https://xgboost.readthedocs.io) · [SHAP](https://shap.readthedocs.io) · [Scikit-Learn](https://scikit-learn.org) · [Pandas](https://pandas.pydata.org).
