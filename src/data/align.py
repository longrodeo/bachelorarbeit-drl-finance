# ---------------------------------------------------------------------------
# Datei: src/data/align.py
# Zweck: Preisreihen auf einen vorgegebenen Handelskalender bringen.
# Hauptfunktionen: ``align_to_trading_days`` für Aktien, ``resample_crypto_last``
#   für Kryptowährungen mit 24/7-Handel.
# Ein-/Ausgabe: ``pd.DataFrame`` mit ``DatetimeIndex`` → reindizierte Serie.
# Abhängigkeiten: ``pandas``; Stolpersteine sind tz-naive Indizes und Lücken im
#   Kalender.
# ---------------------------------------------------------------------------
"""
Hilfsfunktionen zum Ausrichten von Preisserien auf einen Handelskalender.
Enthält Funktionen für klassische Wertpapiere sowie für Kryptowährungen.
Einsatzgebiet: INTERIM-Aufbereitung vor Feature-Berechnung.
Input sind Pandas-DataFrames mit DatetimeIndex (idealerweise UTC).
Abhängigkeiten: ausschließlich ``pandas``; Probleme entstehen oft bei tz-naiven
Indizes oder fehlenden Handelstagen.
"""

import pandas as pd  # Kernbibliothek für Zeitreihen-Manipulation

def align_to_trading_days(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Preisreihen strikt auf vorgegebene Handelstage ausrichten.

    Parameters
    ----------
    df : pd.DataFrame
        Eingangsserie mit DatetimeIndex.
    cal_idx : pd.DatetimeIndex
        Zielindex der Handelstage (tz-aware).

    Returns
    -------
    pd.DataFrame
        Reindizierte Serie, fehlende Tage bleiben ``NaN``.
    """
    if df.index.tz is None:  # falls Datumsindex zeitzonen-naiv ist
        df = df.tz_localize("UTC")  # auf UTC setzen für eindeutige Vergleiche
    return df.reindex(cal_idx)  # harte Reindizierung ohne Füllung

def resample_crypto_last(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Kryptoreihen auf Handelskalender verdichten.

    Idee: Kryptowährungen handeln täglich → zunächst tägliche Frequenz,
    anschließend nur Handelstage behalten.

    Parameters
    ----------
    df : pd.DataFrame
        Zeitreihe auf Minuten/Stundenbasis oder täglicher Frequenz.
    cal_idx : pd.DatetimeIndex
        Zielhandelskalender.

    Returns
    -------
    pd.DataFrame
        Serie mit einem Wert pro Handelstag.
    """
    if df.index.tz is None:  # sicherstellen, dass Index tz-aware ist
        df = df.tz_localize("UTC")
    daily = df.resample("1D").last()  # tägliche Aggregation, unabhängig vom Kalender
    return daily.reindex(cal_idx)  # auf Handelstage ausrichten
