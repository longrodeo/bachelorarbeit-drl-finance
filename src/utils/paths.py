# -----------------------------------------------------------------------------
# Central configuration helper that resolves project paths and asset metadata.
# Loads the YAML specification and exposes convenience helpers for data access.
# These utilities ensure consistent directory creation and lookup across modules.
# -----------------------------------------------------------------------------
# src/utils/paths.py
from __future__ import annotations
from pathlib import Path
import yaml
from typing import Dict, List

# Project base directory and configuration specification file.
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "data_spec.yml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    # SPEC stores the entire data specification loaded from YAML.
    SPEC: dict = yaml.safe_load(f) or {}

# Resolve individual paths from the configuration with sensible fallbacks.
_paths = SPEC.get("paths", {}) or {}
RAW_DIR       = (BASE_DIR / _paths.get("raw", "data/raw")).resolve()
INTERIM_DIR   = (BASE_DIR / _paths.get("interim_dir", "data/interim")).resolve()
CLEAN_DIR     = (BASE_DIR / _paths.get("clean_dir", "data/clean")).resolve()
RAW_DIR.mkdir(parents=True, exist_ok=True)
# INTERIM_DIR.mkdir(parents=True, exist_ok=True)  # Optional: enable to ensure interim directory exists eagerly.
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

INTERIM_PANEL = (BASE_DIR / _paths.get("interim_panel", INTERIM_DIR / "panel.parquet")).resolve()
FEATURES_ASSETS   = (BASE_DIR / _paths.get("clean_panel",   CLEAN_DIR / "features_v1.parquet")).resolve()
RISKFREE_FILE = (BASE_DIR / _paths.get("riskfree",      CLEAN_DIR / "riskfree.parquet")).resolve()
RISKFREE_RAW_FILE = (BASE_DIR / _paths.get("riskfree",      RAW_DIR / "riskfree_raw.parquet")).resolve()
MANIFEST_FILE = (BASE_DIR / _paths.get("manifest_clean", CLEAN_DIR / "_manifest.json")).resolve()

def raw_asset_path(asset: str) -> Path:
    """Return the path to the raw parquet file for a given asset ticker.

    Args:
        asset: Asset identifier used to derive the parquet filename.

    Returns:
        Path pointing to the parquet file within the raw asset directory.
    """

    file_rel = (SPEC.get("assets") or {}).get("file", "assets.yml")
    # folder_name represents the subdirectory derived from the asset config file name.
    folder_name = Path(file_rel).stem
    path = (RAW_DIR / folder_name).resolve()
    # Ensure the directory exists before returning the file path.
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{asset}.parquet"

def _normalize_asset(asset: str) -> str:
    """Normalize crypto tickers by removing dashes and uppercasing the code.

    Args:
        asset: Original asset ticker possibly containing separator characters.

    Returns:
        Uppercase ticker string adjusted to the expected provider format.
    """

    # Convert Yahoo-style tickers (e.g., BTC-USD) to Tiingo-style (BTCUSD).
    if "-" in asset and asset.upper().endswith("-USD"):
        return asset.replace("-", "").upper()
    return asset.upper()

def _load_assets_file(path_rel: str) -> Dict[str, List[str]]:
    """Load a grouped asset YAML file and return ticker groups.

    Args:
        path_rel: Relative path to the YAML file describing asset groups.

    Returns:
        Dictionary mapping group names to ordered lists of tickers.
    """

    file_path = (BASE_DIR / path_rel).resolve()
    with open(file_path, "r", encoding="utf-8") as f:
        # data captures the entire YAML structure for the asset definitions.
        data = yaml.safe_load(f) or {}
    # Keep only dictionary entries that are lists, representing asset groups.
    groups = {k: v for k, v in data.items() if isinstance(v, list)}
    return groups

def get_asset_groups() -> Dict[str, List[str]]:
    """Combine asset groups from configuration files and inline definitions.

    Returns:
        Dictionary mapping asset group names to lists of asset tickers.
    """

    assets_cfg = SPEC.get("assets") or {}
    groups: Dict[str, List[str]] = {}
    # Attempt to load groups from an external file if specified.
    file_rel = assets_cfg.get("file")
    if file_rel:
        groups.update(_load_assets_file(file_rel))
    # Merge inline definitions that can override or extend file-based groups.
    for k, v in (assets_cfg.items()):
        if k == "file":
            continue
        if isinstance(v, list):
            groups[k] = list(v)
    return groups

def get_assets_flat(groups: Dict[str, List[str]] | None = None) -> List[str]:
    """Flatten group definitions into a unique, ordered list of asset tickers.

    Args:
        groups: Optional precomputed group dictionary; defaults to configuration.

    Returns:
        List of asset tickers preserving the first occurrence order.
    """

    if groups is None:
        groups = get_asset_groups()
    out: List[str] = []
    # out accumulates all tickers in order of their group appearance.
    for lst in groups.values():
        out.extend(lst)
    # Enforce uniqueness while preserving the order of first occurrence.
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def get_window():
    """Retrieve the configured start and end dates for the data window.

    Returns:
        Tuple of strings representing the start and end dates.
    """

    win = SPEC.get("window", {}) or {}
    # start and end default to explicit top-level values or window block fallback.
    start = str(SPEC.get("start", win.get("start", "2019-01-01")))
    end   = str(SPEC.get("end",   win.get("end",   "2019-03-31")))
    return start, end

def get_project_root() -> Path:
    """Return the absolute project root directory two levels above this file.

    Returns:
        Path object pointing to the repository root directory.
    """
    return BASE_DIR

ROOT        = get_project_root()
DATA_DIR    = ROOT / "data"
ACCOUNT_DIR = DATA_DIR / "accounting_demo"
CONFIG_DIR  = ROOT / "config"

# Clean data artifacts and accounting output locations.
CLEAN_PANEL  = FEATURES_ASSETS
FEATURES_NORM = DATA_DIR / "clean" / "features_v1_raw_z.parquet"
SNAP_PATH    = ACCOUNT_DIR / "portfolio_snapshots.parquet"
REWARD_PATH  = ACCOUNT_DIR / "rewards_log.parquet"

# State configuration YAML files.
SPEC_S0_YAML = CONFIG_DIR / "state_config" / "state0.yml"
SPEC_S1_YAML = CONFIG_DIR / "state_config" / "state1.yml"

# Output directory for state debugging and visualization artifacts.
OUT_DIR      = DATA_DIR / "states_demo"
