# ---------------------------------------------------------------------------
# Parquet IO utilities that provide robust save/load helpers with engine fallbacks.
# Ensures directories exist before writing and prefers fastparquet when available.
# Serves as a centralized access point for consistent Parquet handling across code.
# ---------------------------------------------------------------------------
from __future__ import annotations  # enable postponed type annotations
from pathlib import Path  # object-oriented filesystem path handling
from typing import Union  # allow str or Path inputs in function signatures
import pandas as pd  # pandas DataFrame serialization utilities

__all__ = ["save_parquet", "load_parquet"]  # exported utility functions

def _ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory of ``path`` exists before writing files.

    Args:
        path: Destination path whose parent directory must be created.
    """

    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Persist a pandas ``DataFrame`` to Parquet with safe engine fallbacks.

    Args:
        df: Table that should be stored on disk.
        path: Target file path where the Parquet artifact will be written.
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
                f"Failed to write Parquet file. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )

def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load a Parquet file using fastparquet with pyarrow as fallback.

    Args:
        path: File path that should be read into a pandas ``DataFrame``.

    Returns:
        DataFrame reconstructed from the stored Parquet file.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    try:
        return pd.read_parquet(p, engine="fastparquet")
    except Exception as e_fast:
        try:
            return pd.read_parquet(p, engine="pyarrow")
        except Exception as e_arrow:
            raise RuntimeError(
                f"Failed to read Parquet file. "
                f"fastparquet: {e_fast}, pyarrow: {e_arrow}"
            )
