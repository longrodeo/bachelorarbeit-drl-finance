import os
from typing import Optional
import requests
import pandas as pd

from src.data.calendar import nyse_trading_days
from src.data.align import align_to_trading_days
# ------------------------- täglicher Zinssatz FED -------------------------

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

def _resolve_fred_api_key(passed: Optional[str] = None) -> str:
    """
    Liefert den FRED API-Key. Reihenfolge:
    1) explizit übergebener Key (Argument)
    2) Umgebungsvariablen: FRED_API_KEY, FRED_API_TOKEN, FRED_KEY
    """
    # key = "Platzhalter"
    key = passed or os.environ.get("FRED_API_KEY") or os.environ.get("FRED_API_TOKEN") or os.environ.get("FRED_KEY")
    if not key:
        raise ValueError("FRED API Key fehlt. Setze FRED_API_KEY (oder übergib api_key=...).")
    return key


def fetch_fred_nyse_daily(
    series_id: str = "DGS3MO",
    start: str = "1990-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    fill: str = "ffill",   # "ffill", "bfill" oder None
    tz: str = "UTC",
) -> pd.Series:
    api_key = _resolve_fred_api_key(api_key)

    if end is None:
        end = pd.Timestamp.today(tz="UTC").date().isoformat()

    # --- FRED abrufen ---
    params = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "file_type": "json",
        "api_key": api_key,
    }
    resp = requests.get(FRED_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    obs = pd.DataFrame(data.get("observations", []))
    if obs.empty:
        return pd.Series(name=series_id, dtype="float64")

    # --- in Series (Prozent p.a.) ---
    obs["value"] = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")
    obs["date"]  = pd.to_datetime(obs["date"], utc=True).dt.tz_localize(None)
    s = pd.Series(obs["value"].values, index=obs["date"].values, name=series_id).sort_index()

    # --- NYSE-Handelstage  + Reindex  ---
    cal_idx = nyse_trading_days(start=s.index.min().date().isoformat(), end=end, tz=tz)   # :contentReference[oaicite:2]{index=2}
    df = align_to_trading_days(s.to_frame(), cal_idx)                                     # :contentReference[oaicite:3]{index=3}

    if fill == "ffill":
        df[series_id] = df[series_id].ffill()
    elif fill == "bfill":
        df[series_id] = df[series_id].bfill()

    return df[series_id]

def annual_pct_to_daily_rate(y_annual_pct: pd.Series, basis: int = 360) -> pd.Series:
    # z.B. DGS3MO (% p.a.) -> tägliche einfache Rate (Dezimal)
    return (y_annual_pct.astype(float) / 100.0) / float(basis)

def daily_factor(y_annual_pct: pd.Series, basis: int = 360) -> pd.Series:
    # (1 + r_d)
    return 1.0 + annual_pct_to_daily_rate(y_annual_pct, basis=basis)