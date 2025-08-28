"""
PyTest-Konfiguration: stellt sicher, dass Projekt- und src-Verzeichnisse im
``sys.path`` liegen, damit Importe auch ohne Installation funktionieren.
"""

# Importiert Betriebssystem- und Systemmodule, um Pfade zu manipulieren und den Python-Suchpfad zu erweitern
import os, sys
# Absoluter Pfad zur Projektwurzel (eine Ebene über dem Tests-Verzeichnis)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Pfad zum Quellcode-Verzeichnis, sodass `import`-Statements funktionieren
SRC  = os.path.join(ROOT, "src")
# Fügt ROOT und SRC dem Modul-Suchpfad hinzu, falls noch nicht vorhanden
for p in (ROOT, SRC):
    # Sicherstellen, dass der Pfad nicht doppelt eingefügt wird
    if p not in sys.path:
        # Einfügen am Anfang priorisiert lokale Pakete gegenüber global installierten
        sys.path.insert(0, p)
