"""
Tests für den NYSE-Handelskalender: stellt sicher, dass der zurückgegebene
Index sortiert, UTC-basiert und wochenendfrei ist.
"""

# pandas liefert Datums-/Zeitindex-Utilities für Assertions
import pandas as pd
# zu testende Funktion, erzeugt NYSE-Handelstage
from src.data.calendar import nyse_trading_days

def test_nyse_trading_days_basic():
    """Kalender liefert korrekte Eigenschaften für Januar 2024."""
    idx = nyse_trading_days(start="2024-01-01", end="2024-01-31")  # Erzeuge Handelskalender für Januar 2024
    # 1) Typ & Sortierung
    assert isinstance(idx, pd.DatetimeIndex)  # Rückgabe muss ein DatetimeIndex sein
    assert idx.is_monotonic_increasing  # und chronologisch sortiert
    # 2) Zeitzone (UTC-aware)
    assert idx.tz is not None and str(idx.tz) == "UTC"  # Zeitzone muss gesetzt und UTC sein
    # 3) Kein Wochenende
    assert all(ts.weekday() < 5 for ts in idx)  # jeder Eintrag repräsentiert einen Werktag (Mo-Fr)
    # 4) Kein leerer Index
    assert len(idx) > 0  # Kalender darf nicht leer sein
