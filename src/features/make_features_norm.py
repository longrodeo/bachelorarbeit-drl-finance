# src/features/make_features_norm.py
import pandas as pd

from utils.paths import CLEAN_PANEL, CLEAN_DIR, BASE_DIR
from utils.parquet_io import load_parquet, save_parquet
from features.obs_norm import rolling_zscore

IN  = CLEAN_PANEL
OUT = CLEAN_DIR / "features_v1_norm.parquet"
WINDOW = 50

print("[PATHS]", "BASE_DIR=", BASE_DIR, "\n         IN =", IN, "\n         OUT=", OUT)
df = load_parquet(IN)  # MultiIndex: (date, asset)
num_cols = df.select_dtypes("number").columns
skip = {"dividends", "stock_splits"}
cols = [c for c in num_cols if c not in skip]

def _per_asset(g: pd.DataFrame) -> pd.DataFrame:
    z = rolling_zscore(g[cols], window=WINDOW)  # nur Vergangenheit ≤ t
    z.columns = cols  # gleiche Namen behalten
    return z

df_norm = df.groupby(level="asset", group_keys=False).apply(_per_asset)
save_parquet(df_norm, OUT)
print("saved ->", OUT)



