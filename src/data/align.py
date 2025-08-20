# src/data/align.py
import pandas as pd

def align_to_trading_days(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Reindiziert beliebige Preisserien (z. B. ETFs) hart auf Handelstage.
    Lässt Lücken bewusst als NaN stehen (keine Fills).
    Erwartet DatetimeIndex (tz-aware, idealerweise UTC).
    """
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.reindex(cal_idx)

def resample_crypto_last(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Krypto (7 Tage/Woche) auf Handelstage herunterbrechen: last-Observation per Handelstag.
    1) auf Tagesfreq downsamplen (last)
    2) auf Handelstage reindizieren (NaN bleiben sichtbar)
    """
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    daily = df.resample("1D").last()  # Kalender-Unabhängig
    return daily.reindex(cal_idx)
