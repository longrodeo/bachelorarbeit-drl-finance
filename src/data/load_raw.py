# src/data/load_raw.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import os, requests
import pandas as pd


from src.utils.parquet_io import save_parquet

# wenn vorhanden
try:
    import exchange_calendars as xcals
    _CAL = xcals.get_calendar("XNYS")
except Exception:
    _CAL = None



# --------------------------- Utilities ---------------------------

def _nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    if _CAL is not None:
        return _CAL.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.bdate_range(start=start, end=end)


def _norm_cols(cols: Iterable) -> list[str]:
    """Spaltennamen in snake_case; bei MultiIndex (tuple) 1. Element nehmen."""
    out = []
    for c in cols:
        if isinstance(c, tuple):
            c = c[0]
        out.append(str(c).lower().replace(" ", "_"))
    return out


# --------------------------- Loader ------------------------------

def _load_tiingo(asset: str, start: str, end: str) -> pd.DataFrame:
    token = os.getenv("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY nicht gesetzt. Bitte als Umgebungsvariable speichern.")

    # --- Asset normalisieren ---
    is_crypto = "-" in asset or asset.lower() in ["btcusd", "ethusd"]
    asset_norm = asset.replace("-", "").lower() if is_crypto else asset

    if is_crypto:
        # ---- Crypto API ----
        url = "https://api.tiingo.com/tiingo/crypto/prices"
        params = {
            "tickers": asset_norm,
            "startDate": start,
            "endDate": end,
            "resampleFreq": "1day",
            "token": token
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data or "priceData" not in data[0]:
            raise ValueError(f"Keine Crypto-Daten von Tiingo für {asset}")
        df = pd.DataFrame(data[0]["priceData"])

    else:
        # ---- Equity/ETF API ----
        url = f"https://api.tiingo.com/tiingo/daily/{asset_norm}/prices"
        params = {
            "token": token,
            "startDate": start,
            "endDate": end,
            "resampleFreq": "daily"
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError(f"Keine Equity-Daten von Tiingo für {asset}")
        df = pd.DataFrame(data)

    # --- Zeitspalte robust normalisieren ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "time" in df.columns:  # fallback bei Crypto
        df["date"] = pd.to_datetime(df["time"])
    else:
        raise KeyError(f"Keine Zeitspalte in Tiingo-Daten gefunden für {asset}")

    df = df.set_index("date")

    # --- Einheitliche Spalten ---
    mapping = {
        "adjClose": "adj_close",
        "divCash": "dividends",
        "splitFactor": "stock_splits"
    }
    df = df.rename(columns=mapping)

    # Crypto hat kein adj_close → setze = close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # fehlende Spalten ergänzen
    for col in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]:
        if col not in df.columns:
            df[col] = 0.0

    return df[["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]]

__all__ = ["download_raw_prices"]

def download_raw_prices(
    asset_list: list[str],
    start: str,
    end: str,
    out_dir: str | Path = "data/raw",
    provider: str = "tiingo",
) -> list[Path]:
    """
    Download-only. Schreibt 1:1 Providerdaten pro Asset unter data/raw/{ASSET}.parquet.
    - Keine Alignments, keine Fills.
    - Index wird auf DatetimeIndex 'date' normiert.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for asset in asset_list:
        if provider != "tiingo":
            raise NotImplementedError("Aktuell nur tiingo implementiert.")
        raw = _load_tiingo(asset, start, end)

        if raw is None or raw.empty:
            print(f"[WARN] Keine Providerdaten für {asset} in {start}–{end}.")
            continue

        # Sicherstellen: DatetimeIndex mit Name 'date'
        if not isinstance(raw.index, pd.DatetimeIndex):
            if "date" in raw.columns:
                raw = raw.set_index("date")
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"

        fpath = outp / f"{asset}.parquet"
        save_parquet(raw, fpath)
        print(f"[OK] RAW gespeichert: {fpath} (rows={len(raw)})")
        written.append(fpath)

    if not written:
        raise RuntimeError("Kein Asset erfolgreich als RAW gespeichert.")
    return written

