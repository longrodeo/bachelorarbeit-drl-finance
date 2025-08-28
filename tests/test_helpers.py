"""
Tests für Hilfsfunktionen wie RNG-Seed und Logger.
Verifizieren reproduzierbare Zufallszahlen sowie korrekt konfiguriertes Logging.
"""

# Diese Datei überprüft generische Utilities, die in mehreren Modulen verwendet
# werden: Setzen des Zufallszahlengenerators sowie Logger-Erstellung.
# Abhängigkeiten: `numpy` für Zufallszahlen, `os`/`tmp_path` für Filesystem.
# Edge-Cases: mehrfaches Seeden muss identische Sequenzen erzeugen, Logger darf
# keine doppelten Handler erstellen.
# Ziel ist es, Regressionen bei Reproduzierbarkeit und Logausgabe zu vermeiden.

# OS-Module für Umgebungsfunktionen, etwa Pfade
import os
# NumPy liefert Zufallszahlen und Vergleichsoperationen
import numpy as np

def test_set_seed_numpy_repro():
    """Mehrfaches Seeden liefert identische Zufallsreihen."""
    from src.utils.helpers import set_seed  # lokale Importvermeidung globaler Seiteneffekte
    set_seed(123)  # RNG initialisieren
    a = np.random.RandomState(0).rand(3)  # separater Generator, sollte unabhängig funktionieren
    # reproduzierbare globale RNG:
    set_seed(123)  # erneut seeden für Vergleichssequenz
    x1 = np.random.rand(5)  # erste Zufallsreihe aus globalem Generator
    set_seed(123)  # seed zurücksetzen
    x2 = np.random.rand(5)  # zweite Zufallsreihe
    assert np.allclose(x1, x2)  # Sequenzen müssen identisch sein

def test_logger_basic(tmp_path):
    """Logger liefert Singleton und schreibt in Datei."""
    from src.utils.helpers import get_logger  # Import der Logger-Hilfsfunktion
    log_file = tmp_path / "run.log"  # temporären Logpfad festlegen
    logger = get_logger("TEST", to_file=str(log_file))  # Logger mit Datei-Handler erzeugen
    logger.info("hello")  # eine Logzeile schreiben
    # Handler nicht doppelt:
    logger2 = get_logger("TEST")  # erneuter Abruf desselben Loggers
    assert logger is logger2  # erwartetes Singleton-Verhalten
    # File wurde beschrieben:
    s = log_file.read_text(encoding="utf-8")  # Dateiinhalt lesen
    assert "hello" in s  # Logzeile muss enthalten sein
