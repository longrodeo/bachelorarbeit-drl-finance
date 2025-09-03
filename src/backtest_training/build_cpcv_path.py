from __future__ import annotations

import os, shutil
from pathlib import Path
from typing import Iterable, Dict, List, Literal

import pandas as pd

from utils.parquet_io import load_parquet, save_parquet
from utils.paths import ROOT, CLEAN_DIR, CONFIG_DIR


# ----------------- 1) Jahres-Parquets erzeugen -----------------
import pandas as pd

def _normalize_datetime_index(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Bringt den Frame in eine Form mit Datums-Index:
    - einfacher DatetimeIndex -> (df_sorted, None)
    - MultiIndex -> (df_sorted, name_der_datums_ebene)
    - Spaltenlayout -> setzt Datums-Spalte zum Index
    Versucht 'date'/'datetime'/'timestamp'/'time' (case-insensitive).
    """
    # 1) einfacher DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index(), None

    # 2) MultiIndex: Datums-Ebene finden oder konvertieren
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        # a) Ebene ist bereits datetime
        for i, name in enumerate(names):
            vals = df.index.get_level_values(i)
            if isinstance(vals, pd.DatetimeIndex):
                return df.sort_index(), name or names[i]
        # b) Ebene mit "date"-ähnlichem Namen -> in datetime konvertieren
        for i, name in enumerate(names):
            if name and name.lower() in {"date","datetime","timestamp","time"}:
                tmp = df.reset_index()
                tmp[name] = pd.to_datetime(tmp[name], utc=True, errors="coerce")
                tmp = tmp.set_index(names).sort_index()
                return tmp, name
        # c) Notlösung: erste Ebene versuchen zu parsen
        first_name = names[0] or "date"
        tmp = df.reset_index()
        tmp[first_name] = pd.to_datetime(tmp[first_name], utc=True, errors="coerce")
        tmp = tmp.set_index(names).sort_index()
        if isinstance(tmp.index.get_level_values(0), pd.DatetimeIndex):
            return tmp, names[0]
        raise ValueError("MultiIndex ohne identifizierbare Datums-Ebene.")

    # 3) Spaltenlayout: geeignete Spalte finden
    for c in ("date","datetime","timestamp","time","Date","Timestamp"):
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
            out = out.set_index(c).sort_index()
            return out, None

    cols = list(df.columns)
    idx_names = list(df.index.names) if hasattr(df.index, "names") else [df.index.name]
    raise ValueError(f"Keine Datums-Spalte/Ebene gefunden. index={idx_names}, columns={cols[:10]}")

def _slice_by_year(df: pd.DataFrame, date_level: str | None, year: int) -> pd.DataFrame:
    """
    Jahres-Slice für einfachen Index oder MultiIndex.
    """
    if isinstance(df.index, pd.MultiIndex):
        if date_level is None:
            raise ValueError("date_level erforderlich für MultiIndex.")
        idx = df.index.get_level_values(date_level)
        return df[idx.year == year]
    else:
        return df[df.index.year == year]


def split_parquets_by_year(
    features_parq: str | Path,
    riskfree_parq: str | Path,
    years_cpcv: Iterable[int] = range(2015, 2021),
    years_wf: Iterable[int] = range(2021, 2025),
    out_cpcv_root: str | Path = "data/cpcv/years",
    out_wf_root: str | Path = "data/walk_forward/years",
) -> Dict[str, List[Path]]:
    feats, feats_date_lvl = _normalize_datetime_index(load_parquet(features_parq))
    rf, rf_date_lvl = _normalize_datetime_index(load_parquet(riskfree_parq))

    created = {"cpcv_features":[], "cpcv_riskfree":[], "wf_features":[], "wf_riskfree":[]}
    out_cpcv_root = Path(out_cpcv_root); out_wf_root = Path(out_wf_root)

    for y in years_cpcv:
        f = _slice_by_year(feats, feats_date_lvl, y)
        r = _slice_by_year(rf, rf_date_lvl, y)
        p_f = out_cpcv_root/"features"/f"{y}.parquet"
        p_r = out_cpcv_root/"riskfree"/f"{y}.parquet"
        save_parquet(f, p_f); save_parquet(r, p_r)
        created["cpcv_features"].append(p_f); created["cpcv_riskfree"].append(p_r)

    for y in years_wf:
        f = _slice_by_year(feats, feats_date_lvl, y)
        r = _slice_by_year(rf, rf_date_lvl, y)
        p_f = out_wf_root/"features"/f"{y}.parquet"
        p_r = out_wf_root/"riskfree"/f"{y}.parquet"
        save_parquet(f, p_f); save_parquet(r, p_r)
        created["wf_features"].append(p_f); created["wf_riskfree"].append(p_r)

    return created

# ----------------- 2) Pfade/Splits aus CSV materialisieren -----------------
def _mat(src: Path, dst: Path, mode: Literal["copy","hardlink","symlink"]="copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try: os.link(src, dst)
        except OSError: shutil.copy2(src, dst)
    else:  # symlink
        try: os.symlink(src, dst)
        except OSError: shutil.copy2(src, dst)

def materialize_cpcv_paths_from_csv(
    path_assignment_csv: str | Path,
    years_features_dir: str | Path = "data/cpcv/years/features",
    years_riskfree_dir: str | Path = "data/cpcv/years/riskfree",
    out_paths_root: str | Path = "data/cpcv/paths",
    mode: Literal["copy","hardlink","symlink"]="copy",
) -> Dict[int, Dict[str, Dict[str, List[Path]]]]:
    """
    Baut Ordner 1:1 gemäß path_assignment.csv:
      - pro path_id -> pro split_id -> 'test' (die 2 Jahre aus der CSV) und 'training' (Komplement).
      - CSV muss Spalten haben: path_id, year_block, split_id
    """
    df = pd.read_csv(path_assignment_csv)
    df["path_id"] = df["path_id"].astype(int)
    df["year_block"] = df["year_block"].astype(int)
    df["split_id"] = df["split_id"].astype(str)

    all_years: List[int] = sorted(df["year_block"].unique().tolist())
    yset = set(all_years)
    plan: Dict[int, Dict[str, Dict[str, List[Path]]]] = {}

    for pid, g in df.groupby("path_id"):
        for sid, sg in g.groupby("split_id"):
            test_years = sorted(sg["year_block"].unique().tolist())
            if len(test_years) != 2:
                raise ValueError(f"path {pid}, split {sid}: erwarte genau 2 Testjahre, gefunden: {test_years}")
            train_years = sorted(yset - set(test_years))

            base = Path(out_paths_root)/f"path_{pid}"/"splits"/sid
            trn_f = base/"training"/"features"; tst_f = base/"test"/"features"
            trn_r = base/"training"/"riskfree";  tst_r = base/"test"/"riskfree"

            created = {"training":{"features":[], "riskfree":[]}, "test":{"features":[], "riskfree":[]}}
            for y in train_years:
                _mat(Path(years_features_dir)/f"{y}.parquet", trn_f/f"{y}.parquet", mode)
                _mat(Path(years_riskfree_dir)/f"{y}.parquet", trn_r/f"{y}.parquet", mode)
                created["training"]["features"].append(trn_f/f"{y}.parquet")
                created["training"]["riskfree"].append(trn_r/f"{y}.parquet")
            for y in test_years:
                _mat(Path(years_features_dir)/f"{y}.parquet", tst_f/f"{y}.parquet", mode)
                _mat(Path(years_riskfree_dir)/f"{y}.parquet", tst_r/f"{y}.parquet", mode)
                created["test"]["features"].append(tst_f/f"{y}.parquet")
                created["test"]["riskfree"].append(tst_r/f"{y}.parquet")

            plan.setdefault(pid, {})[sid] = created

    return plan

# ----------------- CLI / Ablauf -----------------
if __name__ == "__main__":
    

    features_path = CLEAN_DIR / "features_v1.parquet"
    riskfree_path = CLEAN_DIR / "riskfree.parquet"
    path_csv = CONFIG_DIR / "cpcv" / "2015_2020" / "path_assignment_2015_2020.csv"

    out_cpcv_root  = ROOT / "data" / "cpcv" / "years"
    out_wf_root    = ROOT / "data" / "walk_forward" / "years"
    out_paths_root = ROOT / "data" / "cpcv" / "paths"

    # Debug-Ausgabe (hilft sofort beim nächsten Problem)
    print("[paths] using:")
    print("  features :", features_path)
    print("  riskfree :", riskfree_path)
    print("  path_csv :", path_csv)
    print("[out_cpcv]  ", out_cpcv_root)
    print("[out_wf]    ", out_wf_root)
    print("[out_paths] ", out_paths_root)

    assert features_path.is_file(), f"Missing: {features_path}"
    assert riskfree_path.is_file(), f"Missing: {riskfree_path}"
    assert path_csv.is_file(), f"Missing: {path_csv}"

    # 1) Jahres-Parquets aus den Rohdateien schneiden
    created = split_parquets_by_year(
        features_parq=features_path,  # absolute Pathlib-Pfade
        riskfree_parq=riskfree_path,
        years_cpcv=range(2015, 2021),
        years_wf=range(2021, 2025),
        out_cpcv_root=out_cpcv_root,
        out_wf_root=out_wf_root,
    )
    print("[split] erstellt:", {k: len(v) for k, v in created.items()})

    # 2) Pfade/Splits exakt gemäß CSV materialisieren
    plan = materialize_cpcv_paths_from_csv(
        path_assignment_csv=path_csv,
        years_features_dir=out_cpcv_root / "features",
        years_riskfree_dir=out_cpcv_root / "riskfree",
        out_paths_root=out_paths_root,
        mode="copy",  # "hardlink"/"symlink" optional (unter Windows oft Admin nötig)
    )
    print("[paths] fertig aufgebaut. Pfade:", sorted(plan.keys()))
