# -----------------------------------------------------------------------------
# Downloads RAW price data from the Tiingo API for equities and crypto assets
# and persists each symbol as a standalone Parquet file for subsequent pipeline
# stages.
# -----------------------------------------------------------------------------

"""Download RAW market data from Tiingo and persist it under the RAW directory."""

from __future__ import annotations  # enable postponed evaluation of annotations

import os
from typing import Iterable, Optional, List

import pandas as pd
import requests

from src.utils.paths import raw_asset_path, _normalize_asset
from src.utils.parquet_io import save_parquet


def _is_crypto(asset: str) -> bool:
    """Return ``True`` if the ticker looks like a crypto pair (USD suffix)."""
    return asset.upper().endswith("USD")


def _load_tiingo(asset: str, start: str, end: str, token: Optional[str] = None) -> pd.DataFrame:
    """Load raw data for a single asset directly from Tiingo.

    Parameters
    ----------
    asset : str
        Symbol or ticker to request.
    start, end : str
        ISO-formatted start and end dates.
    token : str | None
        API token; defaults to ``TIINGO_API_KEY`` from the environment.

    Returns
    -------
    pd.DataFrame
        Unmodified response payload from the Tiingo API.
    """
    token = token or os.getenv("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY is not set.")

    if _is_crypto(asset):
        url = "https://api.tiingo.com/tiingo/crypto/prices"
        params = {"tickers": asset.lower(), "startDate": start, "endDate": end, "resampleFreq": "1day", "token": token}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not payload:
            raise ValueError(f"No crypto data returned for {asset}.")
        rows = payload[0].get("priceData", [])
        return pd.DataFrame(rows)
    else:
        url = f"https://api.tiingo.com/tiingo/daily/{asset}/prices"
        params = {"startDate": start, "endDate": end, "resampleFreq": "daily", "token": token}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return pd.DataFrame(r.json())


def download_raw_prices(assets: Iterable[str], start: str, end: str, token: Optional[str] = None) -> List[str]:
    """Download multiple assets and write each one to a Parquet file.

    Parameters
    ----------
    assets : Iterable[str]
        Tickers to retrieve from Tiingo.
    start, end : str
        Date range for the request.
    token : str | None
        Optional API token overriding the environment variable.

    Returns
    -------
    List[str]
        File paths of successfully written Parquet outputs.
    """
    written: List[str] = []  # collect successfully written Parquet paths
    for asset in assets:
        norm_asset = _normalize_asset(asset)
        try:
            df = _load_tiingo(norm_asset, start, end, token=token)
        except Exception as e:  # noqa: BLE001 - keep broad for resilience in data scripts
            print(f"[WARN] {asset}: download failed ({e}), skipping.")
            continue
        path = raw_asset_path(asset)
        save_parquet(df, path)
        written.append(str(path))
    return written
