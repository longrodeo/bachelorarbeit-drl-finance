# src/data/load_raw.py
from __future__ import annotations

import os
from typing import Iterable, Optional, List

import pandas as pd
import requests

from src.utils.paths import raw_asset_path, _normalize_asset
from src.utils.parquet_io import save_parquet

def _is_crypto(asset: str) -> bool:
    return asset.upper().endswith("USD")

def _load_tiingo(asset: str, start: str, end: str, token: Optional[str] = None) -> pd.DataFrame:
    """Rohdaten 1:1 von Tiingo (kein Rename/Index-Set)."""
    token = token or os.getenv("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY is not set.")

    if _is_crypto(asset):
        url = "https://api.tiingo.com/tiingo/crypto/prices"
        params = {"tickers": asset.lower(), "startDate": start, "endDate": end, "resampleFreq": "1day", "token": token}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        payload = r.json()
        if not payload:
            raise ValueError(f"No crypto data returned for {asset}.")
        rows = payload[0].get("priceData", [])
        return pd.DataFrame(rows)
    else:
        url = f"https://api.tiingo.com/tiingo/daily/{asset}/prices"
        params = {"startDate": start, "endDate": end, "resampleFreq": "daily", "token": token}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        return pd.DataFrame(r.json())

def download_raw_prices(assets: Iterable[str], start: str, end: str, token: Optional[str] = None) -> List[str]:
    """Schreibt eine Parquet pro Asset (RAW, unverändert)."""
    written: List[str] = []
    for asset in assets:
        norm_asset = _normalize_asset(asset)
        try:
            df = _load_tiingo(norm_asset, start, end, token=token)
        except Exception as e:
            print(f"[WARN] {asset}: konnte nicht geladen werden ({e}), skip.")
            continue
        path = raw_asset_path(asset)
        save_parquet(df, path)  # eigener Helper (fastparquet-first)
        written.append(str(path))
    return written
