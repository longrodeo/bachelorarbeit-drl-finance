# src/features/make_yearly_norm.py
from pathlib import Path
import pandas as pd
import argparse
from src.utils.parquet_io import load_parquet, save_parquet
from src.features.obs_norm import rolling_zscore

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2025)))
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--src", nargs="+", default=["data/cpcv/years/features", "data/walk_forward/years/features"])
    ap.add_argument("--target", default="data/gold/features/v1/panel")
    args = ap.parse_args()

    # >>> Fix: Pfade am Repo-Root ausrichten (src ist Geschwister von data)
    ROOT = Path(__file__).resolve().parents[2]       # .../REPO/
    SRC_ROOTS = [(ROOT / s).resolve() for s in args.src]
    TARGET = (ROOT / args.target).resolve()
    TARGET.mkdir(parents=True, exist_ok=True)
    print("[INFO] REPO ROOT:", ROOT)
    print("[INFO] SEARCH ROOTS:")
    for r in SRC_ROOTS: print("   -", r, "(exists:", r.exists(), ")")
    print("[INFO] TARGET:", TARGET)

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

    summary = []
    for y in args.years:
        raw_path = find_raw(y)
        if raw_path is None:
            print(f"[SKIP] {y}: raw not found in {[str(r) for r in SRC_ROOTS]}")
            continue

        df = load_parquet(raw_path)  # erwartet (date, asset)-Index
        if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {"date", "asset"}:
            if {"date", "asset"}.issubset(df.columns):
                df = df.set_index(["date", "asset"]).sort_index()
            else:
                raise AssertionError(f"{raw_path} hat keinen (date, asset) Index und keine entsprechenden Spalten.")

        num = df.select_dtypes("number").columns
        z = (df.groupby(level="asset", group_keys=False)[num]
               .apply(lambda g: rolling_zscore(g, window=args.window))
               .fillna(0.0))

        panel = df.join(z, how="left", lsuffix="_raw", rsuffix="_norm")
        out = TARGET / f"year={y}.parquet"
        save_parquet(panel, out)

        d = df.index.get_level_values("date")
        summary.append((y, len(df), str(d.min().date()), str(d.max().date()), str(out)))
        print(f"[OK] {y} -> {out}")

    if summary:
        chk = pd.DataFrame(summary, columns=["year","rows","min_date","max_date","path"]).sort_values("year")
        print("\nCHECKLISTE")
        print(chk.to_string(index=False))

