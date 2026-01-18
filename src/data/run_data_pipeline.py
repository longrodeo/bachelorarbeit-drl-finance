# -----------------------------------------------------------------------------
# Command-line entry point that orchestrates RAW→INTERIM→CLEAN processing,
# attaches risk-free rates, and produces walk-forward as well as CPCV
# feature panels for experimentation.
# -----------------------------------------------------------------------------

import argparse
from pathlib import Path
import pandas as pd

from src.utils import paths
from src.utils.parquet_io import load_parquet, save_parquet

from src.data.load_raw import download_raw_prices
from src.data.build_interim import build_interim_prices
from src.data.build_clean import build_clean_data
from src.features.obs_norm import rolling_zscore
from src.features.riskfree_interest import load_and_process_FED_interest


def ensure_raw_interim_clean_rf(day_basis: int = 360) -> pd.DataFrame:
    """Ensure RAW, INTERIM, CLEAN, and risk-free artifacts exist and return CLEAN.

    Parameters
    ----------
    day_basis : int
        Day-count convention used when constructing the risk-free factor.

    Returns
    -------
    pd.DataFrame
        CLEAN feature panel (unnormalised) loaded or freshly generated.
    """
    assets = paths.get_assets_flat()
    start, end = paths.get_window()

    raw_dir = Path(paths.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not any(raw_dir.iterdir()):
        print(f"[RAW] downloading {len(assets)} assets {start}..{end} → {raw_dir}")
        download_raw_prices(assets=assets, start=start, end=end)

    interim_pq = Path(paths.INTERIM_PANEL)
    if not interim_pq.is_file():
        print(f"[INTERIM] build NYSE-aligned panel → {interim_pq}")
        build_interim_prices(assets=assets, start=start, end=end, save=True)

    clean_pq = Path(paths.CLEAN_PANEL)
    if not clean_pq.is_file():
        print(f"[CLEAN] build features → {clean_pq}")
        prices = load_parquet(paths.INTERIM_PANEL)
        panel = build_clean_data(prices, out_path=str(clean_pq))
    else:
        panel = load_parquet(paths.CLEAN_PANEL)

    rf_pq = Path(paths.RISKFREE_FILE)
    rf_pq_raw = Path(paths.RISKFREE_RAW_FILE)
    print(f"[RF] build risk-free rates → {rf_pq}")
    load_and_process_FED_interest(start=start, basis=day_basis, out_path_clean=str(rf_pq), out_path_raw=str(rf_pq_raw))

    return panel


def attach_risk_free(df: pd.DataFrame, day_basis: int = 360) -> pd.DataFrame:
    """Attach risk-free metrics to the feature panel by broadcasting over assets."""
    rf = load_parquet(paths.RISKFREE_FILE)
    if rf is None or rf.empty:
        print(f"[WARN] RF empty → {paths.RISKFREE_FILE}")
        return df

    cols = [c for c in ["risk_free_rate", "rf_daily_rate", "rf_daily_factor"] if c in rf.columns]
    if not cols:
        print("[WARN] RF has no expected columns; skipping join.")
        return df

    dates = df.index.get_level_values("date")
    rf = rf.reindex(dates).ffill().bfill()

    df = df.copy()
    for c in cols:
        df[c] = rf[c].to_numpy()

    return df


def join_raw_and_z(df: pd.DataFrame, window: int, clip: float = 6.0, eps: float = 1e-8) -> pd.DataFrame:
    """Create a combined RAW+Z panel with rolling z-scores per asset.

    Parameters
    ----------
    df : pd.DataFrame
        CLEAN feature panel indexed by (date, asset).
    window : int
        Rolling window size for the z-score computation.
    clip : float
        Z-score clipping threshold to limit extreme values.
    eps : float
        Numerical stabiliser passed to ``rolling_zscore``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ``*_raw`` copies and ``*_z`` normalised columns.
    """
    exclude = {"rf_daily_factor_raw", "dividends", "stock_splits", "is_cash"}  # fields not normalised via z-score

    numeric = df.select_dtypes("number")  # numeric subset used for duplication and scaling
    raw = numeric.add_suffix("_raw")  # copy numeric values with *_raw suffix

    z_src = numeric.drop(columns=[c for c in exclude if c in numeric.columns], errors="ignore")  # subset eligible for z-score

    def _z(group: pd.DataFrame) -> pd.DataFrame:
        try:
            return rolling_zscore(group, window=window, clip=clip, eps=eps)
        except TypeError:
            return rolling_zscore(group, window=window)

    z = z_src.groupby(level="asset", group_keys=False).apply(_z).add_suffix("_z")

    out = raw.join(z, how="left")
    non_num = df.drop(columns=df.select_dtypes("number").columns, errors="ignore")
    if not non_num.empty:
        out = non_num.join(out, how="left")
    return out

def _ensure_synth_schema(prices: pd.DataFrame) -> pd.DataFrame:
    """Ensure minimal OHLCV schema expected by CLEAN builder for synthetic paths.

    The CLEAN builder expects at least: open, high, low, close, adj_open, adj_close,
    volume, dividends, stock_splits (MultiIndex: date, asset).
    Missing columns are filled with conservative defaults.
    """
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != ["date", "asset"]:
        raise ValueError("Synthetic prices must be a MultiIndex with index names ['date','asset'].")

    df = prices.copy()

    # Dividends / splits are required by the CLEAN schema but are typically absent in synthetic files.
    if "dividends" not in df.columns:
        df["dividends"] = 0.0
    if "stock_splits" not in df.columns:
        df["stock_splits"] = 0.0

    # Ensure unadjusted OHLC columns exist (fallback to adjusted where reasonable).
    if "open" not in df.columns and "adj_open" in df.columns:
        df["open"] = df["adj_open"]
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    if "high" not in df.columns and "open" in df.columns and "close" in df.columns:
        df["high"] = df[["open", "close"]].max(axis=1)
    if "low" not in df.columns and "open" in df.columns and "close" in df.columns:
        df["low"] = df[["open", "close"]].min(axis=1)

    # Volume should exist; if missing, set to 0 (features like ADV then become neutral).
    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df


def attach_constant_risk_free(df: pd.DataFrame, rf_annual: float, day_basis: int = 360) -> pd.DataFrame:
    """Attach a constant risk-free path (annual rate) to the panel, broadcast over assets."""
    rf_annual = float(rf_annual)
    daily_rate = rf_annual / float(day_basis)
    daily_factor = 1.0 + daily_rate
    out = df.copy()
    out["risk_free_rate"] = rf_annual
    out["rf_daily_rate"] = daily_rate
    out["rf_daily_factor"] = daily_factor
    return out


def run_synth(synth_dir: Path, window: int, rf_annual: float, day_basis: int) -> None:
    """Build RAW+Z panels for all synthetic scenario parquet files in ``synth_dir``.

    Input:  MultiIndex (date, asset) OHLCV parquet (e.g., data/synth/bear_1y.parquet).
    Output: Same directory with suffix ``_features.parquet``.
    """
    synth_dir = Path(synth_dir)
    if not synth_dir.exists():
        raise FileNotFoundError(f"Synth directory not found: {synth_dir}")

    files = sorted([
        p for p in synth_dir.glob("*.parquet")
        if not p.name.endswith("_features.parquet")
        and not p.name.endswith("_raw_z.parquet")
        and not p.name.endswith("_clean.parquet")
    ])

    if not files:
        print(f"[SYNTH] No parquet files found in {synth_dir}")
        return

    for p in files:
        print(f"[SYNTH] {p.name}: CLEAN→RF(const)→RAW+Z")
        prices = load_parquet(p)
        prices = _ensure_synth_schema(prices)

        clean = build_clean_data(prices, out_path=None)
        clean = attach_constant_risk_free(clean, rf_annual=rf_annual, day_basis=day_basis)
        panel = join_raw_and_z(clean, window=window)

        out = p.with_name(p.stem + "_features.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[WRITE] {out}")
        save_parquet(panel, out)


def run_wf(window: int, out_path: Path, day_basis: int) -> None:
    """Build the walk-forward RAW+Z panel and persist it to ``out_path``."""
    clean = ensure_raw_interim_clean_rf(day_basis=day_basis)
    clean = attach_risk_free(clean, day_basis=day_basis)
    panel = join_raw_and_z(clean, window=window)
    print(f"[WRITE] WF panel RAW+Z → {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(panel, out_path)


def run_cpcv(years: list[int], window: int, base_dir: Path, day_basis: int) -> None:
    """Build CPCV year-specific RAW+Z panels for the provided year list."""
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


def main() -> None:
    """Parse CLI arguments and execute the requested pipeline stages."""
    ap = argparse.ArgumentParser(description="One-button pipeline: RAW→INTERIM→CLEAN→RF→WF & CPCV (RAW+Z)")
    ap.add_argument("--wf_window", type=int, default=60, help="Rolling-Z window for the global WF build")
    ap.add_argument("--cpcv_window", type=int, default=30, help="Rolling-Z window per year for CPCV panels")
    ap.add_argument("--years", nargs="+", type=int, default=[2015, 2016, 2017, 2018, 2019, 2020], help="CPCV years to build")
    ap.add_argument("--day_basis", type=int, default=360, help="Day-count basis for rf_daily_factor_raw")
    ap.add_argument("--skip_wf", action="store_true", help="Skip building the walk-forward dataset")
    ap.add_argument("--skip_cpcv", action="store_true", help="Skip building CPCV yearly datasets")
    # Synthetic scenario builder (optional)
    ap.add_argument("--build_synth", action="store_true",
                    help="Build synthetic scenario feature panels under --synth_dir")
    ap.add_argument("--synth_dir", type=str, default="data/synth",
                    help="Directory containing synthetic scenario parquet files")
    ap.add_argument("--synth_window", type=int, default=60, help="Rolling-Z window for synthetic panels")
    ap.add_argument("--rf_annual", type=float, default=0.04,
                    help="Assumed annual risk-free rate for synthetic scenarios (e.g., 0.04=4%)")

    args = ap.parse_args()

    if args.build_synth:
        run_synth(
            synth_dir=Path(args.synth_dir),
            window=args.synth_window,
            rf_annual=args.rf_annual,
            day_basis=args.day_basis,
        )

    if not args.skip_wf:
        default_out = Path(paths.CLEAN_PANEL).with_name(Path(paths.CLEAN_PANEL).stem + "_raw_z.parquet")
        run_wf(window=args.wf_window, out_path=default_out, day_basis=args.day_basis)

    if not args.skip_cpcv:
        base_dir = Path(paths.CLEAN_DIR) / "cpcv" / "years"
        run_cpcv(years=args.years, window=args.cpcv_window, base_dir=base_dir, day_basis=args.day_basis)


if __name__ == "__main__":
    main()
