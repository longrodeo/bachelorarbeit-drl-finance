# src/data/build_interim.py
from __future__ import annotations
from typing import Optional, Sequence, Set, Dict

import pandas as pd

from src.data.calendar import nyse_trading_days
from src.data.align import align_to_trading_days, resample_crypto_last
from src.utils.paths import INTERIM_PANEL
from src.utils.parquet_io import load_parquet, save_parquet

# Mapping Provider → kanonisch (nur in INTERIM anwenden)
PROVIDER_TO_CANONICAL = {
    "adjclose": "adj_close",
    "divcash": "dividends",
    "splitfactor": "stock_splits",
}

DEFAULT_SPEC: Dict[str, object] = {
    "fields": ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"],
    "require_base_fields": True,
    "base_fields": ["open", "high", "low", "close"],
    "calendar": "XNYS",
}

def _sanitize(s: str) -> str:
    return str(s).strip().replace(" ", "_").replace("-", "_").lower()

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={c: PROVIDER_TO_CANONICAL.get(_sanitize(c), _sanitize(c)) for c in df.columns}, inplace=True)
    return df

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
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
    sessions: Optional[pd.DatetimeIndex] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Aus RAW-Parquets ein Panel (MultiIndex: date, asset) bauen."""
    cfg = dict(DEFAULT_SPEC)
    cal_idx = nyse_trading_days(start, end, tz="UTC")  # tz-aware Index (UTC)
    if spec:
        cfg.update(spec)

    fields = list(cfg.get("fields", []))
    require_base = bool(cfg.get("require_base_fields", True))
    base_fields = set(cfg.get("base_fields", ["open", "high", "low", "close"]))
    crypto_assets = {a.upper() for a in (crypto_assets or set())}

    frames = []
    for asset in assets:
        from utils.paths import raw_asset_path  # lazy import, vermeidet Zyklen
        f = raw_asset_path(asset)
        if not f.exists():
            raise FileNotFoundError(f"RAW file not found: {f}.")
        raw = load_parquet(f)  # RAW unverändert (date ist Spalte)

        # Fenster schneiden (tz-naiv)
        if "date" in raw.columns:
            date = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None)
            start_date = pd.to_datetime(start); end_date = pd.to_datetime(end)
            raw = raw.loc[(date >= start_date) & (date <= end_date)]

        # Normalisieren
        df = _standardize_columns(raw)
        df = _to_datetime_index(df)

        # Feldauswahl
        keep = [c for c in fields if c in df.columns]
        df = df[keep].copy()

        # Basisspalten-Policy
        if require_base:
            missing = [c for c in base_fields if c not in df.columns]
            if missing:
                raise ValueError(f"[{asset}] Missing base fields after mapping: {missing}")

        # sinnvolle Defaults nur falls in fields verlangt
        if "adj_close" in fields and "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        if "dividends" in fields and "dividends" not in df.columns:
            df["dividends"] = 0.0
        if "stock_splits" in fields and "stock_splits" not in df.columns:
            df["stock_splits"] = 1.0
        if "volume" in fields and "volume" not in df.columns:
            df["volume"] = 0.0

        # Align/Downsample
        is_crypto = (asset.upper() in crypto_assets) or asset.upper().endswith("USD")
        if is_crypto:
            df = resample_crypto_last(df, cal_idx)  # 7-Tage-Krypto → Handelstage (last)
        else:
            df = align_to_trading_days(df, cal_idx)  # Equities hart auf Handelstage

        df["asset"] = asset
        frames.append(df)

    out = pd.concat(frames, axis=0)
    out = out.set_index("asset", append=True).sort_index()
    out.index.set_names(["date", "asset"], inplace=True)

    if save:
        save_parquet(out, INTERIM_PANEL)
    return out
