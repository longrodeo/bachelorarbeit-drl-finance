from pathlib import Path
import pandas as pd

from src.utils.parquet_io import load_parquet, save_parquet
# ------------------------------------------------------------
# Kombiniert pro Path alle Folds (laut Manifest) zu je EINEM Parquet
# - Manifest MUSS Spalten 'fold_id' und 'path_id' enthalten.
# - Aus jedem Fold werden test/seg_01 und test/seg_02 gelesen.
# - Es werden pro Path drei Dateien geschrieben:
#     Path_XX/portfolio_snapshots.parquet
#     Path_XX/trade_events.parquet            (wenn vorhanden)
#     Path_XX/rewards.parquet                 (wenn vorhanden; Dateiname im Fold kann variieren)
# - Chronologische Sortierung und Duplikate auf 't' werden entfernt.
# ------------------------------------------------------------

# Welche Rewards-Dateinamen wir akzeptieren (wir nehmen den ersten, der existiert):
REWARD_CANDIDATES = (
    "rewards.parquet",
    "reward_snapshots.parquet",
    "rewards_snapshots.parquet",
)

def _read_if_exists(p: Path) -> pd.DataFrame | None:
    if p.is_file():
        df = load_parquet(p)
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"])
        return df
    return None

def _collect_fold_segment(root: Path, fold_id: int) -> dict[str, list[pd.DataFrame]]:
    """Liest aus fold_{id}/test/seg_01, seg_02 die drei Tabellen (falls vorhanden)."""
    out = {"portfolio": [], "events": [], "rewards": []}
    fold = root / f"fold_{fold_id:02d}" / "test"
    for seg in ("seg_01", "seg_02"):
        base = fold / seg
        # portfolio_snapshots
        df_port = _read_if_exists(base / "portfolio_snapshots.parquet")
        if df_port is None:
            raise FileNotFoundError(f"Fehlt: {base/'portfolio_snapshots.parquet'}")
        out["portfolio"].append(df_port)

        # trade_events (optional)
        df_ev = _read_if_exists(base / "trade_events.parquet")
        if df_ev is not None:
            out["events"].append(df_ev)

        # rewards (optional, mehrere mögliche Dateinamen)
        for cand in REWARD_CANDIDATES:
            df_rw = _read_if_exists(base / cand)
            if df_rw is not None:
                out["rewards"].append(df_rw)
                break  # ersten Treffer nehmen
    return out

def _concat_time_clean(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Einfach: concat, nach t sortieren, doppelte t löschen (keep first)."""
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"])
        df = df.sort_values("t")
        df = df.drop_duplicates(subset="t", keep="first")
    return df.reset_index(drop=True)

def combine_paths(run_dir: str | Path, manifest_csv: str | Path, n_paths: int = 5) -> None:
    """
    Liest Manifest (Spalten: fold_id, path_id) und baut pro Path drei Parquets:
    - Path_XX/portfolio_snapshots.parquet
    - Path_XX/trade_events.parquet
    - Path_XX/rewards.parquet
    """
    run_dir = Path(run_dir)
    mf = pd.read_csv(manifest_csv, engine="python")
    if not {"fold_id", "path_id"}.issubset(mf.columns):
        raise ValueError("Manifest muss Spalten 'fold_id' und 'path_id' enthalten.")
    mf = mf.loc[:, ["fold_id", "path_id"]].copy()
    mf["fold_id"] = mf["fold_id"].astype(int)
    mf["path_id"] = mf["path_id"].astype(int)

    # pro Path die Folds einsammeln
    for p in range(1, n_paths + 1):
        folds = mf.loc[mf["path_id"] == p, "fold_id"].tolist()
        if not folds:
            continue

        port_dfs, ev_dfs, rw_dfs = [], [], []
        for fid in folds:
            collected = _collect_fold_segment(run_dir, fid)
            port_dfs.extend(collected["portfolio"])
            ev_dfs.extend(collected["events"])
            rw_dfs.extend(collected["rewards"])

        # zusammenführen & speichern
        path_dir = run_dir / f"Path_{p:02d}"
        path_dir.mkdir(parents=True, exist_ok=True)

        df_port = _concat_time_clean(port_dfs)
        if not df_port.empty:
            df_port.to_parquet(path_dir / "portfolio_snapshots.parquet")

        df_ev = _concat_time_clean(ev_dfs)
        if not df_ev.empty:
            df_ev.to_parquet(path_dir / "trade_events.parquet")

        df_rw = _concat_time_clean(rw_dfs)
        if not df_rw.empty:
            df_rw.to_parquet(path_dir / "rewards.parquet")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Ordner mit fold_01..fold_15")
    ap.add_argument("--manifest", required=True, help="CSV mit Spalten fold_id,path_id")
    ap.add_argument("--n_paths", type=int, default=5)
    args = ap.parse_args()
    combine_paths(args.run_dir, args.manifest, n_paths=args.n_paths)
