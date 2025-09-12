# scripts/smoke_panel_rf.py
from pathlib import Path
import os

# Repo-Root setzen, damit relative Pfade in load_panel_years stimmen
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

from src.data.load_panel_years import load_panel_years

# → Jahre bei Bedarf hier anpassen:
df = load_panel_years([2015, 2016])

# 1) Index & Spalten
assert {"date","asset"} <= set(df.index.names)
cols = df.columns
obs = [c for c in cols if c.endswith("_norm")]
raw = [c for c in cols if c.endswith("_raw")]
assert obs and raw
assert "risk_free_rate_norm" in cols
assert "rf_daily_factor_raw" in cols

# 2) Observations ohne NaNs (wichtig fürs RL)
assert int(df[obs].isna().sum().sum()) == 0

# 3) Risk-free daily factor zur Datumsachse ausgerichtet & plausibel
dates = df.index.get_level_values("date").unique()
rf_factor = df.groupby(level="date")["rf_daily_factor_raw"].first()
assert len(rf_factor) == len(dates)
assert (rf_factor > 0).all()

print("SMOKE OK")
