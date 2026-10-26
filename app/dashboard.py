"""Environmental Intelligence & Multimodal Remote Sensing Dashboard Wrapper."""
import sys
from pathlib import Path

# Forward to src/app/dashboard.py
dashboard_src = Path(__file__).resolve().parents[1] / "src" / "app" / "dashboard.py"
with open(dashboard_src, "r", encoding="utf-8") as f:
    code = f.read()
exec(code)
