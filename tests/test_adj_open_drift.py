# tests/test_adj_open_drift.py
import pandas as pd
from src.utils.parquet_io import load_parquet

RAW = "data/raw/ohlcv.parquet"  # ggf. anpassen

def test_adj_open_drift():
    df = load_parquet(RAW)  # erwartet MultiIndex (date, asset)
    cols = ["open", "close", "adj_close", "adj_open"]
    assert set(cols).issubset(df.columns), f"Fehlende Spalten: {set(cols)-set(df.columns)}"
    m = df[cols].dropna()
    adj_open_calc = m["open"] * (m["adj_close"] / m["close"])
    max_err = float((adj_open_calc - m["adj_open"]).abs().max())
    assert max_err <= 5e-5, f"AdjOpen drift (max |Δ|={max_err:.2e})"
