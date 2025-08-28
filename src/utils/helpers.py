# ---------------------------------------------------------------------------
# Datei: src/utils/helpers.py
# Zweck: Sammlung kleiner Hilfsfunktionen für deterministische Experimente
#   (Seed-Setzung) und einheitliches Logging.
# Hauptfunktionen: ``set_seed`` und ``get_logger``.
# Abhängigkeiten: Standardbibliotheken ``logging``, ``os``, ``random`` sowie
#   ``numpy`` und optional ``torch``.
# Typische Fehler: fehlender PyTorch-Import oder mehrfaches Hinzufügen von
#   Logger-Handlern.
# ---------------------------------------------------------------------------
from __future__ import annotations  # zukünftige Typ-Hints ohne String-Literale
import logging, os, random  # Logging-Framework, Umgebungsvariablen, Zufall
from typing import Optional  # optionaler Parameter-Typ für Pfadangaben
import numpy as np  # numerische Zufallsgeneratoren

def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """
    Setzt Seeds für Python, NumPy und optional PyTorch.
    Hinweis: PYTHONHASHSEED wirkt formal beim Interpreter-Start; das Setzen hier
    hilft v. a. für ggf. gestartete Subprozesse (und schadet nicht).
    
    Parameters
    ----------
    seed : int
        Basiswert für alle Zufallsgeneratoren.
    deterministic_torch : bool, optional
        Erzwingt deterministisches Verhalten bei ``torch`` (CuDNN).
    """
    # Python/OS
    os.environ["PYTHONHASHSEED"] = str(seed)  # Hashseed für Stabilität setzen
    random.seed(seed)  # Python-eigenen PRNG deterministisch machen

    # NumPy
    np.random.seed(seed)  # Numpy-PRNG auf Seed einstellen

    # Torch (optional)
    try:  # Import kann fehlschlagen, wenn Torch nicht installiert ist
        import torch  # schwere Bibliothek für Deep Learning
        torch.manual_seed(seed)  # CPU-Seeds setzen
        torch.cuda.manual_seed_all(seed)  # GPU-Seeds setzen (alle Geräte)
        if deterministic_torch:  # Option für deterministische CuDNN-Läufe
            torch.backends.cudnn.deterministic = True  # deterministische Algorithmen
            torch.backends.cudnn.benchmark = False  # keine autotune-Heuristik
    except ImportError:
        pass  # Torch noch nicht installiert → ignorieren

def get_logger(
    name: str = "BA",
    level: int = logging.INFO,
    *,
    to_file: Optional[str] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Idempotenter Logger:
    - StreamHandler (STDOUT) wird genau einmal sichergestellt.
    - Optional: FileHandler für `to_file` wird genau einmal (pro Pfad) sichergestellt.
    - Keine doppelten Handler; spätere Aufrufe aktualisieren Level/Formatter.
    
    Parameters
    ----------
    name : str
        Logger-Name.
    level : int
        Logging-Level (z. B. ``logging.INFO``).
    to_file : str | None
        Optionaler Pfad für Logdatei.
    fmt : str
        Format-String für Logausgaben.
    datefmt : str
        Datumsformat der Logausgabe.

    Returns
    -------
    logging.Logger
        Konfiguriertes Logger-Objekt ohne doppelte Handler.
    """
    logger = logging.getLogger(name)  # existierenden oder neuen Logger holen
    logger.setLevel(level)  # Mindestlevel setzen
    logger.propagate = False  # keine Weiterleitung an Root-Logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)  # Format definieren

    # 1) StreamHandler sicherstellen (ohne FileHandler—der ist Subklasse von StreamHandler)
    stream_handlers = [  # Filter existierender StreamHandler ohne FileHandler
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    if not stream_handlers:  # falls keiner vorhanden, neu anlegen
        sh = logging.StreamHandler()  # Ausgabe auf STDOUT
        sh.setFormatter(formatter)  # Format zuweisen
        sh.setLevel(level)  # Level setzen
        logger.addHandler(sh)  # Handler anhängen
    else:  # existierende Handler anpassen
        for h in stream_handlers:
            h.setLevel(level)  # Level aktualisieren
            h.setFormatter(formatter)  # Format aktualisieren

    # 2) FileHandler sicherstellen (nur wenn gewünscht und noch nicht vorhanden für genau diesen Pfad)
    if to_file:  # Logging zusätzlich in Datei schreiben
        path = os.path.abspath(to_file)  # absoluter Pfad für Vergleich
        file_handlers = [  # existierende FileHandler mit identischem Pfad suchen
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
        ]
        if not file_handlers:  # wenn noch keiner existiert, neu anlegen
            fh = logging.FileHandler(path, encoding="utf-8")  # Datei-Handler
            fh.setFormatter(formatter)  # Format zuweisen
            fh.setLevel(level)  # Level setzen
            logger.addHandler(fh)  # Handler hinzufügen
        else:  # vorhandene FileHandler aktualisieren
            for h in file_handlers:
                h.setLevel(level)  # Level aktualisieren
                h.setFormatter(formatter)  # Format aktualisieren

    return logger  # fertig konfiguriertes Logger-Objekt zurückgeben
