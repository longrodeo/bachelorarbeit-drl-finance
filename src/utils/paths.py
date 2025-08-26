# src/utils/paths.py
from __future__ import annotations
from pathlib import Path
import yaml
from typing import Dict, List

# Projektbasis & Config-Datei
BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = BASE_DIR / "config" / "data_spec.yml"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    SPEC: dict = yaml.safe_load(f) or {}

# Pfade aus Config (mit Fallbacks)
_paths = SPEC.get("paths", {}) or {}
RAW_DIR       = (BASE_DIR / _paths.get("raw", "data/raw")).resolve()
INTERIM_DIR   = (BASE_DIR / _paths.get("interim_dir", "data/interim")).resolve()
CLEAN_DIR     = (BASE_DIR / _paths.get("clean_dir", "data/clean")).resolve()
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

INTERIM_PANEL = (BASE_DIR / _paths.get("interim_panel", INTERIM_DIR / "panel.parquet")).resolve()
CLEAN_PANEL   = (BASE_DIR / _paths.get("clean_panel",   CLEAN_DIR / "features_v1.parquet")).resolve()
RISKFREE_FILE = (BASE_DIR / _paths.get("riskfree",      CLEAN_DIR / "riskfree.parquet")).resolve()
MANIFEST_FILE = (BASE_DIR / _paths.get("manifest_clean", CLEAN_DIR / "_manifest.json")).resolve()

def raw_asset_path(asset: str) -> Path:
    file_rel = (SPEC.get("assets") or {}).get("file", "assets.yml")
    folder_name = Path(file_rel).stem   # z.B. "assets_regions"
    path = (RAW_DIR / folder_name).resolve()
    path.mkdir(parents=True, exist_ok=True)   # sorgt dafür, dass der Ordner existiert
    return path / f"{asset}.parquet"

# Ticker normalisieren für Kryptoassets
def _normalize_asset(asset: str) -> str:
    # Yahoo-Style BTC-USD → Tiingo-Style BTCUSD
    if "-" in asset and asset.upper().endswith("-USD"):
        return asset.replace("-", "").upper()
    return asset.upper()

# ---- Assets (immer gruppiert) ----------------------------------------------

def _load_assets_file(path_rel: str) -> Dict[str, List[str]]:
    """Liest eine gruppierte Asset-Datei (z. B. equities, crypto, etfs, fx)."""
    file_path = (BASE_DIR / path_rel).resolve()
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    groups = {k: v for k, v in data.items() if isinstance(v, list)}
    return groups

def get_asset_groups() -> Dict[str, List[str]]:
    """
    Liefert gruppierte Assets aus SPEC:
      assets:
        file: config/assets_regions.yml    # oder
        equities: [ ... ]
        crypto:   [ ... ]
    """
    assets_cfg = SPEC.get("assets") or {}
    groups: Dict[str, List[str]] = {}
    # 1) aus Datei laden?
    file_rel = assets_cfg.get("file")
    if file_rel:
        groups.update(_load_assets_file(file_rel))
    # 2) inline-Gruppen mergen (überschreiben/ergänzen Datei)
    for k, v in (assets_cfg.items()):
        if k == "file":
            continue
        if isinstance(v, list):
            groups[k] = list(v)
    return groups

def get_assets_flat(groups: Dict[str, List[str]] | None = None) -> List[str]:
    """Flacht die Gruppen zu einer geordneten, eindeutigen Tickerliste ab."""
    if groups is None:
        groups = get_asset_groups()
    out: List[str] = []
    for lst in groups.values():
        out.extend(lst)
    # Eindeutig bei Erhalt der Reihenfolge
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

# Zeitfenster
def get_window():
    win = SPEC.get("window", {}) or {}
    start = str(SPEC.get("start", win.get("start", "2019-01-01")))
    end   = str(SPEC.get("end",   win.get("end",   "2019-03-31")))
    return start, end
