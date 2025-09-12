# src/data/load_panel.py
from pathlib import Path
import pandas as pd

from src.utils.parquet_io import load_parquet
# Repo-Root unabhängig vom cwd bestimmen (…/REPO)
ROOT = Path(__file__).resolve().parents[2]
CPCV = ROOT / "data/clean/cpcv/years"  # hier liegen die year=YYYY.parquet

def load_panel_years(years: list[int]) -> pd.DataFrame:
    dfs = []
    for y in years:
        p = GOLD / f"{y}_features.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Fehlt: {p} (cwd={Path.cwd()})")
        dfs.append(load_parquet(p))
    df = pd.concat(dfs).sort_index()
    assert df.index.is_unique
    return df
