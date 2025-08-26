# sanity_checks.py
from src.utils.paths import INTERIM_PANEL, RISKFREE_FILE, SPEC, get_window
from src.utils.parquet_io import load_parquet
from src.data.calendar import nyse_trading_days

# 1) Kalender-Konsistenz: INTERIM == NYSE Sessions
start, end = get_window()
cal_idx = nyse_trading_days(start, end, tz="UTC")  # eure Quelle
panel = load_parquet(INTERIM_PANEL)
dates = panel.index.get_level_values("date").unique().sort_values()
assert len(dates) == len(cal_idx), f"Mismatch: {len(dates)} vs {len(cal_idx)} Handelstage"
assert (dates == cal_idx.tz_convert(None)).all(), "INTERIM weicht vom NYSE-Kalender ab"

# 2) TZ-Sauberkeit: alles tz-naiv auf Panel-Ebene
assert dates.tz is None, "INTERIM-Date-Level ist nicht tz-naiv"

# 3) Risk-free Alignment: exakt gleiche Sessions
rf = load_parquet(RISKFREE_FILE).squeeze("columns")
rf_idx = rf.index.sort_values()
assert len(rf_idx) == len(cal_idx), "Risk-free hat andere Anzahl Handelstage"
assert (rf_idx == cal_idx.tz_convert(None)).all(), "Risk-free nicht exakt auf NYSE-Sessions ausgerichtet"

print("✔ Kalender/Alignment passt: INTERIM & Risk-free sind konsistent.")
