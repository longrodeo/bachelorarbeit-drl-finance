from pathlib import Path
import pandas as pd
import argparse

from src.data.build_clean import build_clean_data
from src.utils.parquet_io import load_parquet


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["Bachelorarbeit/data/cpcv/years/features", "Bachelorarbeit/data/walk_forward/years/features"])
    ap.add_argument("--target", default="Bachelorarbeit/data/gold/v2")

    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[2]  # .../REPO/
    SRC_ROOTS = [(ROOT / s).resolve() for s in args.src]
    TARGET = (ROOT / args.target).resolve()
    TARGET.mkdir(parents=True, exist_ok=True)
    print("[INFO] REPO ROOT:", ROOT)
    print("[INFO] SEARCH ROOTS:")
    for r in SRC_ROOTS: print("   -", r, "(exists:", r.exists(), ")")
    print("[INFO] TARGET:", TARGET)

    years = list(range(2015, 2025))

    def find_raw(year: int) -> Path | None:
        # Rekursiv suchen, norm/panel ausschließen
        cands = []
        for base in SRC_ROOTS:
            if not base.exists():
                continue
            for p in base.rglob(f"*{year}*.parquet"):
                name = p.name.lower()
                if any(x in name for x in ["_norm", "panel", "zscore"]):
                    continue
                cands.append(p)
        if not cands:
            return None
        cands.sort(key=lambda p: (not p.name.lower().startswith("features"),
                                  "raw" not in p.name.lower(),
                                  len(p.name)))
        return cands[0]

    for y in years:
        raw_path = find_raw(y)
        if raw_path is None:
            print(f"[SKIP] {y}: raw not found in {[str(r) for r in SRC_ROOTS]}")
            continue

        df = load_parquet(raw_path)

        out = TARGET / f"year={y}.parquet"

        build_clean_data(prices=df, cs_sample_length=2, out_path=out)