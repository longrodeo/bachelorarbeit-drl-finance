# utils/manifest.py
import json, hashlib, os, sys, platform, subprocess
import pandas as pd

def sha256_file(path: str, chunk_size: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def file_summary(path: str) -> dict:
    # Für Parquet: Zeilen/Spalten/Datumsscope grob ermitteln
    try:
        df = pd.read_parquet(path)
        out = {
            "path": path,
            "sha256": sha256_file(path),
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
        }
        if "date" in getattr(df.index, "names", []) or "date" in df.columns:
            d = df.reset_index()
            dates = d["date"] if "date" in d else d.set_index(df.index.names)["date"]
            out["date_min"] = str(pd.to_datetime(dates.min()).date())
            out["date_max"] = str(pd.to_datetime(dates.max()).date())
        return out
    except Exception:
        return {"path": path, "sha256": sha256_file(path)}

def current_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def write_manifest(payload: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
