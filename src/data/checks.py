# ---------------------------------------------------------------------------
# Datei: src/data/checks.py
# Zweck: Mini-Validierungen für DataFrames und Indexe, um fehlerhafte Rohdaten
#   frühzeitig abzufangen.
# Hauptfunktionen: ``assert_no_dupes`` entdeckt doppelte Zeitstempel,
#   ``assert_non_negative`` prüft Wertebereiche, ``report_gaps`` meldet
#   fehlende Sessions.
# Ein-/Ausgabe: pandas-Objekte; Rückgabe meist ``None`` oder Liste von Timestamps.
# Abhängigkeiten: nur ``pandas``; typische Fehler sind NaNs, Negative oder
#   nicht eindeutige Indizes.
# ---------------------------------------------------------------------------
"""
Kleine Validierungsfunktionen für Preis- und Kalenderdaten.
Genutzt in der Datenpipeline, um inkonsistente Indizes oder unerwartete Werte
frühzeitig zu entdecken. Abhängigkeit: ``pandas``.
Fehlerquellen: doppelte Zeitstempel, negative Preise oder fehlende Handelstage.
"""

import pandas as pd  # grundlegende Datenstrukturen

def assert_no_dupes(index: pd.Index) -> None:
    """Prüfen, ob ein Index eindeutige Einträge besitzt."""
    if not index.is_unique:  # Pandas-Flag für eindeutige Labels
        dups = index[index.duplicated()].unique()  # zeigt betroffene Zeitstempel
        raise AssertionError(f"Doppelte Zeitstempel gefunden: {len(dups)}")  # sofort abbrechen

def assert_non_negative(df: pd.DataFrame, cols=None) -> None:
    """Sicherstellen, dass ausgewählte Spalten keine negativen Werte enthalten."""
    sub = df if cols is None else df[cols]  # auf relevante Spalten filtern
    if (sub < 0).any().any():  # Elementweise Prüfung auf Werte < 0
        bad = sub[(sub < 0).any(axis=1)]  # betroffene Zeilen herausfiltern
        raise AssertionError(f"Negative Werte gefunden (erste Zeile: {bad.index[0]})")

def report_gaps(index: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Fehlende Handelstage im Index melden.

    Returns eine Liste der Sessions, die nicht im vorhandenen Index vorkommen.
    """
    missing = sessions.difference(index)  # set-ähnliche Differenzmenge
    return list(missing)  # für einfache Weiterverarbeitung in Tests
