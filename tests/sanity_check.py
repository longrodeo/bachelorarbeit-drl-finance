"""
Einfache Sanity-Checks auf bereits erzeugte Daten.
Vergleicht INTERIM-Panel und Riskfree-Serie mit dem NYSE-Handelskalender und
prüft Zeitzonen-Konsistenz.
"""

# Pfadkonstanten und Fenster-Funktion für den Testzeitraum laden
from src.utils.paths import INTERIM_PANEL, RISKFREE_FILE, SPEC, get_window
# Hilfsfunktion zum Einlesen von Parquet-Dateien
from src.utils.parquet_io import load_parquet
# Handelskalender-Funktion für NYSE-Trading-Days
from src.data.calendar import nyse_trading_days

# 1) Kalender-Konsistenz: INTERIM == NYSE Sessions
start, end = get_window()  # Lese den global definierten Auswertungszeitraum aus den Spezifikationen
cal_idx = nyse_trading_days(start, end, tz="UTC")  # Erzeuge erwarteten Kalenderindex in UTC
panel = load_parquet(INTERIM_PANEL)  # Lade das vorbereitete Panel als DataFrame
dates = panel.index.get_level_values("date").unique().sort_values()  # Extrahiere und sortiere die vorhandenen Handelstage
assert len(dates) == len(cal_idx), f"Mismatch: {len(dates)} vs {len(cal_idx)} Handelstage"  # Anzahl der Tage muss übereinstimmen
assert (dates == cal_idx.tz_convert(None)).all(), "INTERIM weicht vom NYSE-Kalender ab"  # Jeder Tag muss exakt im Kalender enthalten sein

# 2) TZ-Sauberkeit: alles tz-naiv auf Panel-Ebene
assert dates.tz is None, "INTERIM-Date-Level ist nicht tz-naiv"  # Index darf keine Zeitzone tragen

# 3) Risk-free Alignment: exakt gleiche Sessions
rf = load_parquet(RISKFREE_FILE).squeeze("columns")  # Zinsserie laden und DataFrame -> Series konvertieren
rf_idx = rf.index.sort_values()  # vorhandene Handelstage der Zinsserie sortieren
assert len(rf_idx) == len(cal_idx), "Risk-free hat andere Anzahl Handelstage"  # Anzahl muss identisch sein
assert (rf_idx == cal_idx.tz_convert(None)).all(), "Risk-free nicht exakt auf NYSE-Sessions ausgerichtet"  # Jeder Tag muss deckungsgleich sein

print("✔ Kalender/Alignment passt: INTERIM & Risk-free sind konsistent.")  # Nutzerfeedback im Erfolgsfall
