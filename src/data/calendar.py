# ---------------------------------------------------------------------------
# Datei: src/data/calendar.py
# Zweck: Bestimmung von Handelstagen der NYSE als Zeitbasis für Datenpipelines.
# Hauptfunktion: ``nyse_trading_days`` erzeugt ``pd.DatetimeIndex`` zwischen zwei
#   Daten.
# Abhängigkeiten: optional ``exchange_calendars`` (mit Feiertagen) ansonsten
#   Fallback auf ``pandas``-Werktage.
# Edge Cases: falsche Zeitzonen oder fehlende Kalenderbibliothek.
# ---------------------------------------------------------------------------
"""
Bereitstellung eines Handelskalenders der NYSE.
Kernfunktion ``nyse_trading_days`` liefert einen DatetimeIndex aller Sessions.
Verwendet nach Möglichkeit ``exchange_calendars``; fällt sonst auf einfache
Werktage zurück (Feiertage werden dann nicht berücksichtigt).
Typische Fehler: Zeitzonen-Verwechslungen oder fehlende Kalenderbibliothek.
"""

from datetime import datetime  # für Default-Enddatum (heute)
import pandas as pd  # Index- und Zeitreihenoperationen

# Versuch, die spezialisierte Kalenderbibliothek zu laden
try:
    import exchange_calendars as xcals  # externer Kalender mit Feiertagen
    _CAL_LIB = "exchange_calendars"  # Kennzeichen: Bibliothek verfügbar
except ImportError:  # falls Import scheitert, auf Fallback verweisen
    _CAL_LIB = None  # spätere Funktion verwendet einfache Werktage

def nyse_trading_days(start="2000-01-01", end=None, tz="UTC") -> pd.DatetimeIndex:
    """Handelstage der NYSE zwischen zwei Daten bestimmen.

    Parameters
    ----------
    start : str
        Startdatum im ISO-Format.
    end : str | None
        Enddatum; ``None`` bedeutet "heute".
    tz : str
        Zielzeitzone der Ausgabe (z. B. ``"UTC"``).

    Returns
    -------
    pd.DatetimeIndex
        Aufsteigend sortierte, zeitzonenbewusste Handelstage.
    """
    end = end or datetime.utcnow().date().isoformat()  # Default: aktuelles Datum
    if _CAL_LIB == "exchange_calendars":  # Pfad: spezialisierte Bibliothek verfügbar
        cal = xcals.get_calendar("XNYS")  # Kalenderobjekt für NYSE anfordern
        # schedule ist ein DataFrame mit market_open & market_close
        sched = cal.schedule.loc[start:end]  # nur gewünschter Zeitraum
        # Index enthält die Handelstage → auf gewünschte tz normieren
        days = pd.DatetimeIndex(
            sched.index.tz_localize("America/New_York").tz_convert(tz).normalize()
        )
        return days.unique().sort_values()  # Duplikate entfernen, sortieren
    else:  # Fallback: keine Bibliothek vorhanden
        # Fallback: einfache Werktage; NYSE-Feiertage fehlen
        return pd.date_range(start=start, end=end, freq="B", tz=tz)

if __name__ == "__main__":  # kleine Demo bei direktem Aufruf
    idx = nyse_trading_days(start="2019-01-01")  # Handelstage ab 2019 ziehen
    print("Anzahl Handelstage seit 2019-01-01:", len(idx))  # Anzahl ausgeben
    print("Erste 5:", idx[:5].tolist())  # Beispieldaten zeigen
