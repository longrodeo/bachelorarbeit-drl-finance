# src/utils/daten_laden_und_umbauen.py
from __future__ import annotations
import sys

from src.utils.paths import (
    SPEC, get_asset_groups, get_assets_flat, get_window,
    RAW_DIR, INTERIM_DIR, CLEAN_DIR,
    INTERIM_PANEL, CLEAN_PANEL, RISKFREE_FILE, MANIFEST_FILE
)
from src.data.load_raw import download_raw_prices
from src.data.build_interim import build_interim_prices
from src.data.build_clean import build_clean_data, write_clean_manifest
from src.features.riskfree_interest import fetch_fred_nyse_daily
from src.utils.parquet_io import save_parquet

def main():
    groups = get_asset_groups()              # {'equities':[...], 'crypto':[...], ...}
    assets_flat = get_assets_flat(groups)    # flache Liste für Interim/Clean
    start, end = get_window()

    print(f"[1/5] RAW → {RAW_DIR} | Assets={assets_flat} | {start}..{end}")
    download_raw_prices(assets_flat, start, end)

    print(f"[2/5] INTERIM → {INTERIM_PANEL}")
    cryptos = set(groups.get("crypto", []))
    panel_interim = build_interim_prices(
        assets=assets_flat, start=start, end=end, spec=SPEC, crypto_assets=cryptos, save=True
    )

    sessions = panel_interim.index.get_level_values("date").unique().sort_values()

    print(f"[3/5] Risk-free (FRED) → {RISKFREE_FILE}")
    rf_cfg = SPEC.get("risk_free", {}) or {}
    fred_series = rf_cfg.get("series_id", "DGS3MO")
    fred_fill   = rf_cfg.get("fill", "ffill")
    rf_annual_pct = fetch_fred_nyse_daily(series_id=fred_series, start=start, end=end, fill=fred_fill)
    rf_annual = (rf_annual_pct.astype(float) / 100.0).reindex(sessions).ffill().rename("risk_free_annual")
    save_parquet(rf_annual.to_frame(), RISKFREE_FILE)

    print(f"[4/5] CLEAN/Features → {CLEAN_PANEL}")
    features = build_clean_data(
        prices=panel_interim,
        risk_free_annual=rf_annual,   # annualisiert (dezimal), as-of-t
        out_path=str(CLEAN_PANEL),
        cash_symbol=(SPEC.get("cash", {}) or {}).get("symbol", "CASH"),
        cs_sample_length=int((SPEC.get("cs", {}) or {}).get("sample_length", 1)),
    )

    print(f"[5/5] Manifest → {MANIFEST_FILE}")
    try:
        write_clean_manifest(
            spec=SPEC,
            interim_path=str(INTERIM_PANEL),
            macro_path=f"FRED:{fred_series}",
            out_path=str(CLEAN_PANEL),
            manifest_path=str(MANIFEST_FILE),
        )
    except Exception as e:
        print(f"[WARN] Manifest nicht geschrieben: {e}")

    print("\n=== PIPELINE OK ===")
    print("INTERIM:", panel_interim.shape, "| CLEAN:", features.shape)

if __name__ == "__main__":
    sys.exit(main())
