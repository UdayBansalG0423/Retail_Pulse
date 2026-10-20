"""
run.py
------
One-command launcher for RetailPulse.
Validates dependencies, checks data integrity, and launches the Streamlit platform.

Usage:
    python run.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    print("\n" + "=" * 60)
    print("  🛍️  RetailPulse — AI-Powered Retail Analytics Platform")
    print("=" * 60 + "\n")

    # 1. Check data files
    data_file = ROOT / "data" / "processed" / "transactions_clean.csv"
    if not data_file.exists():
        print("⚠️  Processed datasets not detected.")
        print("   Running automated setup pipeline (this takes ~1-2 min)...")
        res = subprocess.run([sys.executable, str(ROOT / "setup_pipeline.py")])
        if res.returncode != 0:
            print("❌ Pipeline setup failed. Please check error messages above.")
            sys.exit(1)
    else:
        print("✅ Processed datasets and machine learning models ready.")

    # 2. Launch Streamlit
    print("\n🚀 Starting RetailPulse Dashboard on http://localhost:8501 ...")
    print("   Press Ctrl+C to stop the server.\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "dashboard" / "app.py"),
        "--server.port=8501",
        "--server.headless=false",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n🛑 RetailPulse dashboard stopped. Goodbye!")


if __name__ == "__main__":
    main()
