# simple_pipeline.py
from __future__ import annotations
from pathlib import Path
import os, sys
import yaml

# ---- eure Pipeline-Bausteine (nur hintereinander aufgerufen) ----
from src.data.load_raw import download_raw_prices
from src.data.build_interim import build_interim_prices, write_interim_manifest
from src.data.build_clean import build_clean_data, write_clean_manifest
from src.features.riskfree_interest import fetch_fred_nyse_daily
from src.utils.parquet_io import save_parquet


fred_key = os.getenv("FRED_API_KEY") or os.getenv("FRED_API_TOKEN") or os.getenv("FRED_KEY")

print("Interpreter:", sys.executable)
print("FRED_API_KEY present?", bool(os.getenv("FRED_API_KEY")))
# ================== KONFIG (einfach oben eintragen) ==================
ASSETS_FILES = ["assets_compact.yml", "assets_regions.yml"]  # eine oder beide
REGIONS = None  # z.B. ["US","EU"] wenn du assets_regions.yml nach Regionen filtern willst, sonst None
START = "2015-01-01"
END   = "2025-08-21"

RAW_DIR      = "data/raw"
INTERIM_PATH = "data/interim/panel.parquet"
CLEAN_PATH   = "data/clean/features_v1.parquet"
MACRO_PATH   = "data/macro/risk_free.parquet"
CASH_SYMBOL  = "CASH"

RISKFREE_SERIES   = "DFF"  # oder "SOFR"
DAY_COUNT         = 360
CS_SAMPLE_LENGTH  = 1      # Corwin–Schultz-Spanne (1–2)

# ================== kleine Helfer ==================
def load_assets(paths, regions=None):
    assets = set()
    regions = set(r.lower() for r in regions) if regions else None
    for p in paths:
        if not Path(p).exists():
            continue
        y = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        if isinstance(y, list):
            assets |= {str(a).strip() for a in y}
        elif isinstance(y, dict):
            if "assets" in y and isinstance(y["assets"], list):
                assets |= {str(a).strip() for a in y["assets"]}
            # regionen-syntax unterstützen
            for k, v in y.items():
                if isinstance(v, list) and (regions is None or k.lower() in regions):
                    assets |= {str(a).strip() for a in v}
    assets = {a for a in assets if a}
    if not assets:
        raise SystemExit("Keine Assets aus den YAMLs gelesen.")
    return sorted(assets)

def detect_crypto(assets):
    out = set()
    for t in assets:
        u = t.upper()
        if "-" in u or u.endswith("USD"):
            out.add(t)
    return out

# ================== Pipeline, der Reihe nach ==================
def main():
    # env-checks
    if "TIINGO_API_KEY" not in os.environ:
        raise SystemExit("TIINGO_API_KEY fehlt.")
    if "FRED_API_KEY" not in os.environ:
        raise SystemExit("FRED_API_KEY fehlt.")

    assets = load_assets(ASSETS_FILES, REGIONS)
    crypto_assets = detect_crypto(assets)

    # 1) RAW
    raw_files = download_raw_prices(
        asset_list=assets,
        start=START,
        end=END,
        out_dir=RAW_DIR,
        provider="tiingo",
    )

    # 2) INTERIM
    interim_spec = {
        "fields": ["open","high","low","close","adj_close","volume","dividends","stock_splits"],
        "align": {"ffill_crypto": False, "calendar": "XNYS"},
        "interim_dataset_id": "panel_v1",
    }
    panel = build_interim_prices(
        asset_list=assets,
        start=START,
        end=END,
        spec=interim_spec,
        raw_dir=RAW_DIR,
        out_path=INTERIM_PATH,
        crypto_assets=crypto_assets,
    )
    # optional Manifest
    write_interim_manifest(
        spec=interim_spec,
        raw_files=[str(p) for p in raw_files],
        out_path=INTERIM_PATH,
        manifest_path="data/interim/_manifest.json",
    )

    # 3) MACRO (Risk-free)
    rf_annual = fetch_fred_nyse_daily(
        series_id=RISKFREE_SERIES,  # z.B. "DFF"
        start=START,
        end=END,
        api_key=fred_key,  # <— HIER EXPLIZIT REIN
        fill="ffill",
        tz="UTC",
    )

    Path(MACRO_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_parquet(rf_annual.to_frame(name=RISKFREE_SERIES), MACRO_PATH)

    # 4) CLEAN (Features + CASH)
    feature_panel = build_clean_data(
        prices=panel,
        risk_free_annual=rf_annual,
        out_path=CLEAN_PATH,
        cash_symbol=CASH_SYMBOL,
        cs_sample_length=CS_SAMPLE_LENGTH,
    )
    # optional Manifest
    clean_spec = {
        "feature_version": "v1",
        "align": {"calendar": "XNYS"},
        "windows": {"adv":20,"sma":[20,60],"ema":[12,26],"rsi":14,"macd":[12,26,9],"boll":[20,2.0],"cci":20,"adx":14},
        "cs": {"sample_length": CS_SAMPLE_LENGTH},
        "risk_free": {"series_id": RISKFREE_SERIES, "day_count": DAY_COUNT},
        "cash": {"symbol": CASH_SYMBOL},
    }
    write_clean_manifest(
        spec=clean_spec,
        interim_path=INTERIM_PATH,
        macro_path=MACRO_PATH,
        out_path=CLEAN_PATH,
        manifest_path="data/clean/_manifest.json",
    )

    print("[DONE] RAW → INTERIM → CLEAN abgeschlossen.")
    print(f"  RAW dir   : {RAW_DIR}")
    print(f"  INTERIM   : {INTERIM_PATH}")
    print(f"  MACRO     : {MACRO_PATH}")
    print(f"  CLEAN     : {CLEAN_PATH}")

if __name__ == "__main__":
    main()
