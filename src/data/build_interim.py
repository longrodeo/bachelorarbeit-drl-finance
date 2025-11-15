# -----------------------------------------------------------------------------
# Combines RAW price Parquets into a harmonised INTERIM panel aligned to the
# NYSE trading calendar, handling column normalisation, timezone cleanup, and
# crypto downsampling ahead of CLEAN feature generation.
# -----------------------------------------------------------------------------

"""Assemble a calendar-aligned price panel from previously downloaded RAW data."""

from __future__ import annotations  # allow forward references in type annotations
from typing import Optional, Sequence, Set, Dict

import pandas as pd

from src.data.trading_calendar import nyse_trading_days
from src.data.align import align_to_trading_days, resample_crypto_last
from src.utils.paths import INTERIM_PANEL
from src.utils.parquet_io import load_parquet, save_parquet

PROVIDER_TO_CANONICAL = {
    "adjclose": "adj_close",
    "adjopen": "adj_open",
    "divcash": "dividends",
    "splitfactor": "stock_splits",
}

DEFAULT_SPEC: Dict[str, object] = {
    "fields": ["open", "high", "low", "close", "adj_open", "adj_close", "volume", "dividends", "stock_splits"],
    "require_base_fields": True,
    "base_fields": ["open", "high", "low", "close"],
    "calendar": "XNYS",
}


def _sanitize(name: str) -> str:
    """Normalise provider-specific column names to lowercase snake case."""
    return str(name).strip().replace(" ", "_").replace("-", "_").lower()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map provider column names to the canonical INTERIM schema."""
    df = df.copy()
    df.rename(columns={c: PROVIDER_TO_CANONICAL.get(_sanitize(c), _sanitize(c)) for c in df.columns}, inplace=True)
    return df


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a "date" column into a tz-naive DateTimeIndex."""
    df = df.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.drop(columns=["date"])
        df.index = dt
        df.index.name = "date"
    return df


def build_interim_prices(
    assets: Sequence[str],
    start: str,
    end: str,
    spec: Optional[dict] = None,
    crypto_assets: Optional[Set[str]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Combine RAW Parquets into a calendar-aligned (date, asset) panel.

    Parameters
    ----------
    assets : Sequence[str]
        Symbols to include in the panel.
    start, end : str
        Inclusive time window delimiting the output range.
    spec : dict | None
        Optional configuration overriding default field selection.
    crypto_assets : Set[str] | None
        Tickers that should be downsampled as crypto instruments.
    save : bool
        Whether to persist the resulting panel to ``INTERIM_PANEL``.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame indexed by (date, asset).
    """
    cfg = dict(DEFAULT_SPEC)
    cal_idx = nyse_trading_days(start, end, tz="UTC")
    if spec:
        cfg.update(spec)

    fields = list(cfg.get("fields", []))  # target column selection for INTERIM
    require_base = bool(cfg.get("require_base_fields", True))  # enforce OHLC availability
    base_fields = set(cfg.get("base_fields", ["open", "high", "low", "close"]))  # canonical OHLC fields
    crypto_assets = {a.upper() for a in (crypto_assets or set())}  # normalised crypto ticker set

    frames = []  # collected DataFrames per asset before concatenation
    for asset in assets:
        from utils.paths import raw_asset_path  # local import to avoid circular dependency

        f = raw_asset_path(asset)
        if not f.exists():
            raise FileNotFoundError(f"RAW file not found: {f}.")
        raw = load_parquet(f)  # load RAW Parquet (date column still present)

        if "date" in raw.columns:
            date = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None)
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            raw = raw.loc[(date >= start_date) & (date <= end_date)]

        df = _standardize_columns(raw)  # unify column names
        df = _to_datetime_index(df)  # promote date column to index

        keep = [c for c in fields if c in df.columns]  # retain only requested fields
        df = df[keep].copy()

        if require_base:
            missing = [c for c in base_fields if c not in df.columns]
            if missing:
                raise ValueError(f"[{asset}] Missing base fields after mapping: {missing}")

        if "adj_close" in fields and "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        if "adj_open" in fields and "adj_open" not in df.columns and "open" in df.columns:
            df["adj_open"] = df["open"]
        if "dividends" in fields and "dividends" not in df.columns:
            df["dividends"] = 0.0
        if "stock_splits" in fields and "stock_splits" not in df.columns:
            df["stock_splits"] = 1.0
        if "volume" in fields and "volume" not in df.columns:
            df["volume"] = 0.0

        is_crypto = (asset.upper() in crypto_assets) or asset.upper().endswith("USD")  # heuristic crypto detection
        if is_crypto:
            df = resample_crypto_last(df, cal_idx)
        else:
            df = align_to_trading_days(df, cal_idx)

        df["asset"] = asset
        frames.append(df)

    out = pd.concat(frames, axis=0)
    out = out.set_index("asset", append=True).sort_index()
    out.index.set_names(["date", "asset"], inplace=True)

    if save:
        save_parquet(out, INTERIM_PANEL)
    return out
