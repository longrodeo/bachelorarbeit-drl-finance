from __future__ import annotations
from pathlib import Path
from typing import Union
import pandas as pd

__all__ = ["save_parquet", "load_parquet"]

def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories for the given path if they do not exist."""
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Speichert ein pandas DataFrame als Parquet-Datei mit stabiler Engine-Auswahl.
    - Erst fastparquet probieren, sonst pyarrow.
    - Index wird standardmäßig gespeichert.
    """
    p = Path(path)
    _ensure_parent_dir(p)
    try:
        df.to_parquet(p, engine="fastparquet")
    except Exception as e_fast:
        try:
            df.to_parquet(p, engine="pyarrow")
        except Exception as e_arrow:
            raise RuntimeError(
                f"Parquet speichern fehlgeschlagen. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Lädt eine Parquet-Datei stabil (fastparquet bevorzugt, sonst pyarrow).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Parquet-Datei nicht gefunden: {p}")
    try:
        return pd.read_parquet(p, engine="fastparquet")
    except Exception as e_fast:
        try:
            return pd.read_parquet(p, engine="pyarrow")
        except Exception as e_arrow:
            raise RuntimeError(
                f"Parquet laden fehlgeschlagen. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )
