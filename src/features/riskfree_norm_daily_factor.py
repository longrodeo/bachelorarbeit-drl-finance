# src/features/make_riskfree_norm.py
import pandas as pd
from utils.parquet_io import load_parquet, save_parquet
from utils.paths import RISKFREE_FILE, CLEAN_DIR
from features.riskfree_interest import daily_factor
from features.obs_norm import rolling_zscore

COL = "risk_free_annual"   # so heißt sie in deinem state_test
WINDOW = 252               # nimm deinen bisherigen Wert; wichtig: >= 10

rf = load_parquet(RISKFREE_FILE).sort_index()
rf[COL] = pd.to_numeric(rf[COL], errors="coerce")

# als DataFrame übergeben (kein Series), kein groupby nötig
z = rolling_zscore(rf[[COL]], window=WINDOW)        # liefert DF mit gleicher Indexierung
out_col = f"{COL}_z"
z.columns = [out_col]

# Warm-up-NaNs auffüllen (g_scalars hat keine Maske)
z[out_col] = z[out_col].fillna(0.0)

 # save_parquet(z, CLEAN_DIR / f"riskfree_norm_z{WINDOW}.parquet")
df = load_parquet(RISKFREE_FILE).sort_index()
df["daily_factor_360"] = daily_factor(df[COL], basis=360)
df_out = z.join(df, how="left")
save_parquet(df_out, CLEAN_DIR / f"riskfree_norm.parquet")
print("saved ->", CLEAN_DIR / f"riskfree_norm.parquet")