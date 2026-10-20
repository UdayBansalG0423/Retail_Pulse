#!/bin/sh
set -e

echo "============================================================"
echo "🛍️  RetailPulse — AI-Powered Retail Analytics Platform"
echo "============================================================"

# Auto-hydration check: if processed dataset missing, run setup
if [ ! -f "data/processed/transactions_clean.csv" ]; then
    echo "⚠️  Processed transaction data not found. Auto-running pipeline setup..."
    python setup_pipeline.py
else
    echo "✅ Processed datasets and ML models verified."
fi

echo "🚀 Starting Streamlit production server on http://0.0.0.0:8501..."
exec streamlit run dashboard/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
