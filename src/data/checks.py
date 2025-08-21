# src/data/checks.py
import pandas as pd

def assert_no_dupes(index: pd.Index) -> None:
    """Sicherstellen, dass Index unique ist."""
    if not index.is_unique:
        dups = index[index.duplicated()].unique()
        raise AssertionError(f"Doppelte Zeitstempel gefunden: {len(dups)}")

def assert_non_negative(df: pd.DataFrame, cols=None) -> None:
    """Keine negativen Werte in Preisfeldern."""
    sub = df if cols is None else df[cols]
    if (sub < 0).any().any():
        bad = sub[(sub < 0).any(axis=1)]
        raise AssertionError(f"Negative Werte gefunden (erste Zeile: {bad.index[0]})")

def report_gaps(index: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Fehlende Handelstage im Index melden."""
    missing = sessions.difference(index)
    return list(missing)
