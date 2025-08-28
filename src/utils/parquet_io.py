# ---------------------------------------------------------------------------
# Datei: src/utils/parquet_io.py
# Zweck: Robuste Ein-/Ausgabefunktionen für Parquet-Dateien mit Fallback auf
#   unterschiedliche Engines.
# Hauptfunktionen: ``save_parquet`` und ``load_parquet``.
# Abhängigkeiten: ``pandas`` sowie ``pathlib`` für Pfadmanipulation.
# Edge Cases: fehlende fastparquet/pyarrow-Installation oder nicht existente
#   Verzeichnisse.
# ---------------------------------------------------------------------------
from __future__ import annotations  # zukünftige Typ-Hints ermöglichen
from pathlib import Path  # objektorientierte Pfadbehandlung
from typing import Union  # Union für Pfadtypen (str/Path)
import pandas as pd  # DataFrame-IO

__all__ = ["save_parquet", "load_parquet"]  # Exportierte Funktionen

def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories for the given path if they do not exist."""
    parent = path.parent  # Elternverzeichnis bestimmen
    if parent and not parent.exists():  # nur bei fehlendem Verzeichnis aktiv
        parent.mkdir(parents=True, exist_ok=True)  # rekursiv anlegen

def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Speichert ein pandas DataFrame als Parquet-Datei mit stabiler Engine-Auswahl.
    - Erst fastparquet probieren, sonst pyarrow.
    - Index wird standardmäßig gespeichert.
    
    Parameters
    ----------
    df : pd.DataFrame
        Zu speichernde Tabelle.
    path : str | Path
        Zieldatei.
    """
    p = Path(path)  # Pfadobjekt erzeugen
    _ensure_parent_dir(p)  # sicherstellen, dass Verzeichnis existiert
    try:  # bevorzugte Engine fastparquet
        df.to_parquet(p, engine="fastparquet")  # schreiben
    except Exception as e_fast:  # Fallback auf pyarrow
        try:
            df.to_parquet(p, engine="pyarrow")  # alternative Engine
        except Exception as e_arrow:  # beide fehlgeschlagen → Fehler melden
            raise RuntimeError(
                f"Parquet speichern fehlgeschlagen. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Lädt eine Parquet-Datei stabil (fastparquet bevorzugt, sonst pyarrow).

    Parameters
    ----------
    path : str | Path
        Dateipfad der zu ladenden Parquet-Datei.

    Returns
    -------
    pd.DataFrame
        Eingelesene Tabelle.
    """
    p = Path(path)  # Pfadobjekt erzeugen
    if not p.is_file():  # Existenzcheck
        raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {p}")
    try:  # bevorzugte Engine fastparquet
        return pd.read_parquet(p, engine="fastparquet")
    except Exception as e_fast:  # Fallback auf pyarrow
        try:
            return pd.read_parquet(p, engine="pyarrow")
        except Exception as e_arrow:  # beide fehlgeschlagen
            raise RuntimeError(
                f"Parquet laden fehlgeschlagen. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )
