# ---------------------------------------------------------------------------
# Manifest helpers that summarize files, compute hashes, and capture git state.
# Designed to provide reproducible metadata snapshots for datasets and artifacts.
# Integrates Parquet inspection, checksum calculation, and JSON serialization.
# ---------------------------------------------------------------------------
import json, hashlib, os, sys, platform, subprocess  # IO helpers and system info
import pandas as pd  # tabular data inspection utilities

def sha256_file(path: str, chunk_size: int = 1<<20) -> str:
    """Compute a streaming SHA256 checksum for the given file path.

    Args:
        path: File path whose contents should be hashed.
        chunk_size: Size of the chunks read from disk to limit memory usage.

    Returns:
        Hexadecimal digest representing the file's SHA256 checksum.
    """

    h = hashlib.sha256()  # incremental hash accumulator
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):  # iterate over file in chunks
            h.update(chunk)
    return h.hexdigest()

def file_summary(path: str) -> dict:
    """Produce a concise metadata summary for the given file.

    Args:
        path: File path to inspect, typically pointing to a Parquet dataset.

    Returns:
        Dictionary containing checksums and, if possible, shape and date metadata.
    """

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
    """Retrieve the short git commit hash for the current working tree.

    Returns:
        Seven-character hash string or ``"unknown"`` when git is unavailable.
    """

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def write_manifest(payload: dict, out_path: str):
    """Serialize the provided manifest dictionary as a JSON file on disk.

    Args:
        payload: Metadata dictionary to persist.
        out_path: Destination file path where the manifest should be saved.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
