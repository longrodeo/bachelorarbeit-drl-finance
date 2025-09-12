# src/pipeline/run_all.py
# Ein Befehl für: RAW → INTERIM → CLEAN → RF → WF-Norm (global) → CPCV (jahresrein, raw+z)
import argparse
from pathlib import Path
import pandas as pd

from src.utils import paths
from src.utils.parquet_io import load_parquet, save_parquet

# vorhandene Bausteine
from src.data.load_raw import download_raw_prices                       # zieht Tiingo-Daten
from src.data.build_interim import build_interim_prices                 # NYSE-align
from src.data.build_clean import build_clean_data                       # Features (kausal, unskaliert)
from src.features.obs_norm import rolling_zscore                        # Z-Score
from src.features.riskfree_interest import load_and_process_FED_interest # Riskfree-Builder


def ensure_raw_interim_clean_rf(day_basis: int = 360) -> pd.DataFrame:
    """Baut RAW→INTERIM→CLEAN (+RF-Datei), falls nötig. Gibt CLEAN (unnormiert) zurück."""
    # 1) Konfig
    assets = paths.get_assets_flat()
    start, end = paths.get_window()

    # 2) RAW (pro Asset-Parquet) — nur ziehen, wenn noch nicht vorhanden/leer
    raw_dir = Path(paths.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not any(raw_dir.iterdir()):
        print(f"[RAW] downloading {len(assets)} assets {start}..{end} → {raw_dir}")
        download_raw_prices(assets=assets, start=start, end=end)

    # 3) INTERIM (Panel, NYSE aligned)
    interim_pq = Path(paths.INTERIM_PANEL)
    if not interim_pq.is_file():
        print(f"[INTERIM] build NYSE-aligned panel → {interim_pq}")
        build_interim_prices(assets=assets, start=start, end=end, save=True)

    # 4) CLEAN (Features unskaliert, kausal)
    clean_pq = Path(paths.CLEAN_PANEL)
    if not clean_pq.is_file():
        print(f"[CLEAN] build features → {clean_pq}")
        prices = load_parquet(paths.INTERIM_PANEL)
        panel = build_clean_data(prices, out_path=str(clean_pq))
    else:
        panel = load_parquet(paths.CLEAN_PANEL)

    # 5) Risk-free Datei bauen (optional, falls Builder vorhanden)
    rf_pq = Path(paths.RISKFREE_FILE)
    rf_pq_raw = Path(paths.RISKFREE_RAW_FILE)
    print(f"[RF] build risk-free rates → {rf_pq}")
    load_and_process_FED_interest(start=start,  basis=360, out_path_clean=str(rf_pq), out_path_raw = str(rf_pq_raw))

    return panel


def attach_risk_free(df: pd.DataFrame, day_basis: int = 360) -> pd.DataFrame:
    rf = load_parquet(paths.RISKFREE_FILE)  # enthält schon risk_free_rate / rf_daily_rate / rf_daily_factor_raw
    if rf is None or rf.empty:
        print(f"[WARN] RF empty → {paths.RISKFREE_FILE}")
        return df

    # gewünschte RF-Spalten übernehmen (nur was vorhanden ist)
    cols = [c for c in ["risk_free_rate", "rf_daily_rate", "rf_daily_factor"] if c in rf.columns]
    if not cols:
        print("[WARN] RF has no expected columns; skipping join.")
        return df

    dates = df.index.get_level_values("date")
    rf = rf.reindex(dates).ffill().bfill()  # auf Panel-Daten ausrichten

    df = df.copy()
    for c in cols:
        df[c] = rf[c].to_numpy()  # broadcast auf alle Assets für dasselbe Datum

    return df



def join_raw_and_z(df: pd.DataFrame, window: int, clip: float = 6.0, eps: float = 1e-8) -> pd.DataFrame:
    """
    Erstellt RAW+Z in einem Frame:
      - Alle numerischen Spalten → *_raw
      - Z-Score über alle numerischen Spalten außer: rf_daily_factor_raw, dividends, stock_splits, is_cash
      - Spread bleibt NICHT ausgeschlossen (wird z-normalisiert)
    """
    EXCLUDE = {"rf_daily_factor_raw", "dividends", "stock_splits", "is_cash"}

    # numerische Rohwerte
    numeric = df.select_dtypes("number")
    raw = numeric.add_suffix("_raw")

    # Z-Score nur auf erlaubten Spalten
    z_src = numeric.drop(columns=[c for c in EXCLUDE if c in numeric.columns], errors="ignore")

    # gruppenweise (asset) rolling z-score; Signatur rolling_zscore(window, clip, eps) ODER mit min_periods
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        try:
            return rolling_zscore(g, window=window, clip=clip, eps=eps)
        except TypeError:
            # Fallback, falls deine Funktion min_periods erwartet/zulässt:
            return rolling_zscore(g, window=window)  # nutze defaults aus deiner Implementierung

    z = z_src.groupby(level="asset", group_keys=False).apply(_z).add_suffix("_z")

    # join: RAW + Z
    out = raw.join(z, how="left")
    # nicht-numerische Spalten (falls vorhanden) wieder anfügen
    non_num = df.drop(columns=df.select_dtypes("number").columns, errors="ignore")
    if not non_num.empty:
        out = non_num.join(out, how="left")
    return out


def run_wf(window: int, out_path: Path, day_basis: int) -> None:
    clean = ensure_raw_interim_clean_rf(day_basis=day_basis)
    clean = attach_risk_free(clean, day_basis=day_basis)
    panel = join_raw_and_z(clean, window=window)
    print(f"[WRITE] WF panel RAW+Z → {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(panel, out_path)


def run_cpcv(years: list[int], window: int, base_dir: Path, day_basis: int) -> None:
    assets = paths.get_assets_flat()
    for y in years:
        start, end = f"{y}-01-01", f"{y}-12-31"
        print(f"[CPCV] Year {y}: INTERIM {start}..{end}")
        interim = build_interim_prices(assets=assets, start=start, end=end, save=False)

        print(f"[CPCV] Year {y}: CLEAN")
        clean = build_clean_data(interim, out_path=None)

        clean = attach_risk_free(clean, day_basis=day_basis)

        panel = join_raw_and_z(clean, window=window)

        out = base_dir / str(y) / f"{y}_features.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WRITE] {out}")
        save_parquet(panel, out)


def main():
    ap = argparse.ArgumentParser(description="One-button pipeline: RAW→INTERIM→CLEAN→RF→WF & CPCV (RAW+Z)")
    ap.add_argument("--wf_window", type=int, default=60, help="Rolling-Z window für WF (global)")
    ap.add_argument("--cpcv_window", type=int, default=30, help="Rolling-Z window pro Jahr (CPCV)")
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018, 2019, 2020], help="CPCV Jahre")
    ap.add_argument("--day_basis", type=int, default=360, help="Zinstagesbasis für rf_daily_factor_raw")
    ap.add_argument("--skip_wf", action="store_true", help="WF-Build überspringen")
    ap.add_argument("--skip_cpcv", action="store_true", help="CPCV-Build überspringen")

    args = ap.parse_args()

    # WF: eine Datei mit RAW+Z über den gesamten Zeitraum
    if not args.skip_wf:
        default_out = Path(paths.CLEAN_PANEL).with_name(Path(paths.CLEAN_PANEL).stem + "_raw_z.parquet")
        run_wf(window=args.wf_window, out_path=default_out, day_basis=args.day_basis)

    # CPCV: je Jahr eine Datei mit RAW+Z (jahresrein)
    if not args.skip_cpcv:
        base_dir = Path(paths.CLEAN_DIR) / "cpcv" / "years"
        run_cpcv(years=args.years, window=args.cpcv_window, base_dir=base_dir, day_basis=args.day_basis)


if __name__ == "__main__":
    main()
