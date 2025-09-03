# src/features/obs_norm.py
import pandas as pd

def rolling_zscore(df: pd.DataFrame, window: int = 252, clip: float = 5.0, eps: float = 1e-8) -> pd.DataFrame:
    """
    Leak-freie Feature-Normalisierung für Finance:
    - rolling Mittel & Std über 'window' Handelstage (nur Vergangenheitsinfos ≤ t)
    - z-Score pro Spalte
    - Winsorizing per clip gegen Ausreißer
    Für den ersten Run bewusst simpel (z-Score). Später optional robust (Median/MAD).
    """
    minp = max(10, window // 4)
    mu = df.rolling(window, min_periods=minp).mean()
    sigma = df.rolling(window, min_periods=minp).std(ddof=0)
    z = (df - mu) / (sigma + eps)
    return z.clip(-clip, clip)
