# -----------------------------------------------------------------------------
# Utilities for loading CLEAN feature panels either by specific CPCV years or as
# a full walk-forward dataset from the data/clean directory structure.
# -----------------------------------------------------------------------------

from pathlib import Path
import pandas as pd

from src.utils.parquet_io import load_parquet

ROOT = Path(__file__).resolve().parents[2]  # repository root independent of cwd
CLEAN_ROOT = ROOT / "data" / "clean"


def load_panel_years(years: list[int]) -> pd.DataFrame:
    """Load CPCV feature panels for the provided list of years.

    Parameters
    ----------
    years : list[int]
        Year identifiers that correspond to ``data/clean/cpcv/years/<year>``.

    Returns
    -------
    pd.DataFrame
        Concatenated panel indexed by (date, asset) for the requested years.
    """
    dfs = []  # accumulate yearly feature panels before concatenation
    for y in years:
        p = CLEAN_ROOT / "cpcv" / "years" / f"{y}" / f"{y}_features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset: {p} (cwd={Path.cwd()})")
        dfs.append(load_parquet(p))
    df = pd.concat(dfs).sort_index()
    assert df.index.is_unique
    return df


def load_panel_wf(features_source: str) -> pd.DataFrame:
    """Load a full walk-forward feature panel by file stem.

    Parameters
    ----------
    features_source : str
        Stem of the Parquet file inside ``data/clean`` (e.g., ``"features_v1_raw_z"``).

    Returns
    -------
    pd.DataFrame
        Complete panel sorted by index, ready for temporal slicing.
    """
    p = CLEAN_ROOT / f"{features_source}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset: {p} (cwd={Path.cwd()})")
    df = load_parquet(p).sort_index()
    assert df.index.is_unique
    return df
