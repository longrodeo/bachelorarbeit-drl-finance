# src/features/make_yearly_norm.py
import pandas as pd
from pathlib import Path
from src.utils.paths import CLEAN_DIR
from src.utils.parquet_io import load_parquet, save_parquet
from src.features.obs_norm import rolling_zscore

YEARS = list(range(2015, 2025))   # 2015…2024
WINDOW = 50
SKIP = {"dividends", "stock_splits"}

for y in YEARS:
    raw_path  = CLEAN_DIR / f"{y}.parquet"
    out_norm  = CLEAN_DIR / f"norm_{y}.parquet"
    out_panel = CLEAN_DIR / f"features_v1_panel_{y}.parquet"

    df = load_parquet(raw_path)  # MultiIndex: (date, asset)
    num_cols = df.select_dtypes("number").columns
    cols = [c for c in num_cols if c not in SKIP]

    # pro Asset: Rolling-Z nur rückwärts, Reset an Jahresgrenze → keine Cross-Year-Fenster
    z = df.groupby(level="asset", group_keys=False)[cols].apply(
        lambda g: rolling_zscore(g, window=WINDOW)
    )

    # RL-Observations brauchen endliche Werte (Warmup): NaNs → 0.0
    z = z.fillna(0.0)

    # getrennt speichern (norm) …
    save_parquet(z, out_norm)

    # … und optional gleich "raw + norm" nebeneinander (für Loader bequem)
    panel = df.join(z, how="left", lsuffix="_raw", rsuffix="_norm")
    save_parquet(panel, out_panel)

    print(f"[OK] {y}: saved ->", out_norm, "and", out_panel)
