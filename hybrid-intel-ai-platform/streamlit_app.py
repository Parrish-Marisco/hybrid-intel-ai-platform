"""
Streamlit Cloud entry point.

Streamlit Cloud's default convention is to run `streamlit_app.py` at the
repo root. The actual dashboard lives at app/hybrid_intel_dashboard_from_json.py;
this file just executes it so the default convention works without
configuration.
"""

from pathlib import Path

DASHBOARD = Path(__file__).resolve().parent / "app" / "hybrid_intel_dashboard_from_json.py"

with open(DASHBOARD, "r", encoding="utf-8") as f:
    exec(compile(f.read(), str(DASHBOARD), "exec"))
