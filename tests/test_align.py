"""
Tests für Ausrichtungsfunktionen: prüfen, ob Zeitreihen korrekt auf
NYSE-Handelstage abgebildet werden und Krypto-Serien auf die letzte Beobachtung
pro Handelstag reduziert werden.
"""

# Diese Testdatei verifiziert die korrekte Ausrichtung von Preiszeitreihen
# auf einen Handelskalender, hier NYSE.
# Sie nutzt pandas für DataFrame-Operationen und setzt voraus, dass die
# Hilfsfunktionen `align_to_trading_days` sowie `resample_crypto_last`
# korrekt mit tz-aware (zeitzonenbewussten) Indizes umgehen.
# Typische Edge-Cases: fehlende Handelstage, Wochenenden und Index-Lücken.
# Ziel ist die Sicherstellung, dass Serien auf genau die im Kalender
# definierten Handelstage reindexiert bzw. verdichtet werden.

# Import von pandas für tabellarische Zeitreihenstrukturen
import pandas as pd
# Handelskalender-Funktion zur Generierung der NYSE-Trading-Days
from src.data.calendar import nyse_trading_days
# Zu testende Funktionen: Reindexing und Resampling für Krypto
from src.data.align import align_to_trading_days, resample_crypto_last

def test_align_basic():
    """ETF-Daten reindizieren und Krypto auf Handelstage verdichten."""
    # Handelskalender für 1.-15. Januar 2024 generieren
    cal = nyse_trading_days(start="2024-01-01", end="2024-01-15")
    # ETF-Reihe: nur zwei Handelstage verfügbar (2. und 5. Januar)
    etf = pd.DataFrame({"Adj Close":[100,101]},
                       index=pd.to_datetime(["2024-01-02","2024-01-05"], utc=True))
    # Reindiziere ETF auf vollständigen Kalender; fehlende Tage → NaN
    out = align_to_trading_days(etf, cal)
    # Erwartung: Index deckt alle Handelstage ab und vorhandene Werte bleiben
    assert out.index.equals(cal) and out.loc["2024-01-02"].notna().all()
    # Crypto-Serie mit täglicher Frequenz inkl. Wochenende definieren
    crypto = pd.DataFrame({"close":[1,2,3,4,5,6,7]},
                          index=pd.to_datetime(["2023-12-31","2024-01-01","2024-01-02",
                                               "2024-01-03","2024-01-04","2024-01-06","2024-01-07"], utc=True))
    # Aggregiere auf Handelstage mittels last-observation-per-day
    out2 = resample_crypto_last(crypto, cal)
    # Index der aggregierten Serie muss ebenfalls dem Kalender entsprechen
    assert out2.index.equals(cal)
