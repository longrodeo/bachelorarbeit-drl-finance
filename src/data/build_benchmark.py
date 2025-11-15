# -----------------------------------------------------------------------------
# Utility script that downloads the ACWI benchmark if missing and rebuilds the
# INTERIM panel for ACWI, BTC-USD, and gold ETF reference assets for consistency.
# Serves as a reproducible entry point when baseline market data needs refreshes.
# -----------------------------------------------------------------------------

from datetime import date
import os

# 1) Project helpers
from src.utils.paths import INTERIM_PANEL  # target path for INTERIM outputs
# 2) RAW loader (Tiingo)
from load_raw import download_raw_prices  # downloads and stores RAW Parquet files :contentReference[oaicite:0]{index=0}
# 3) INTERIM builder (alignment + calendar handling)
from build_interim import build_interim_prices  # assembles (date, asset) panel & saves to INTERIM_PANEL :contentReference[oaicite:1]{index=1}

# ---------------------------------------------
# BENCHMARK DEFINITION
ASSETS = ["ACWI", "BTC-USD", "IAU"]   # adjust gold ETF ticker if necessary
CRYPTO = {"BTC-USD"}                  # ensures proper downsampling to trading days

# TIME WINDOW
START = "2015-01-01"
END   = "2024-12-31"     # adjust horizon as required

# OPTIONAL: ensure TIINGO_API_KEY is present for ACWI downloads
# os.environ["TIINGO_API_KEY"] = "<YOUR_KEY>"  # uncomment if not set in the environment

# ---------------------------------------------
# Step 1: download RAW files only for assets that might be missing (here ACWI)
# BTC & GLD are assumed to exist already; adapt the list to your environment.
RAW_TO_FETCH = ["ACWI"]

print(f"[INFO] Loading RAW data for: {RAW_TO_FETCH}")
_ = download_raw_prices(RAW_TO_FETCH, START, END)  # writes Parquet files under RAW/… :contentReference[oaicite:2]{index=2}

# Step 2: build and persist the INTERIM panel
# - Harmonises core fields (open/high/low/close, adjusted prices, volume, dividends, stock splits)
# - Aligns to NYSE sessions while crypto is resampled to those days (last observation)
# - Saves the result to src.utils.paths.INTERIM_PANEL
print(f"[INFO] Building INTERIM panel for: {ASSETS}")
panel = build_interim_prices(
    assets=ASSETS,
    start=START,
    end=END,
    spec={
        "fields": ["open", "high", "low", "close", "adj_open", "adj_close", "volume", "dividends", "stock_splits"],
        "require_base_fields": True
    },
    crypto_assets=CRYPTO,
    save=True
)  # uses trading calendar alignment helpers internally :contentReference[oaicite:3]{index=3}

print(f"[OK] INTERIM written to: {INTERIM_PANEL}")
print(panel.head())
