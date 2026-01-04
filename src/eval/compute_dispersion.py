from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _stats_table(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    x = df[metric_cols].replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=metric_cols)
    out["mean"] = x.mean(skipna=True)
    out["std"] = x.std(skipna=True, ddof=1)
    out["median"] = x.median(skipna=True)
    out["q25"] = x.quantile(0.25, numeric_only=True)
    out["q75"] = x.quantile(0.75, numeric_only=True)
    out["iqr"] = out["q75"] - out["q25"]
    out["min"] = x.min(skipna=True)
    out["max"] = x.max(skipna=True)
    return out.reset_index(names="metric")


def main(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    rep = run_dir / "_report"

    f_paths = rep / "metrics_per_path.csv"
    f_years = rep / "metrics_per_path_year.csv"

    if not f_paths.is_file():
        raise FileNotFoundError(f"Fehlt: {f_paths}")
    if not f_years.is_file():
        raise FileNotFoundError(f"Fehlt: {f_years}")

    df_paths = pd.read_csv(f_paths)
    df_years = pd.read_csv(f_years)

    # --- 1) Dispersion über Paths (1..5) ---
    # alles numerische außer IDs
    exclude_paths = {"run", "path_id"}
    metric_cols_paths = [c for c in df_paths.columns if c not in exclude_paths]
    metric_cols_paths = [c for c in metric_cols_paths if pd.api.types.is_numeric_dtype(df_paths[c])]

    disp_paths = _stats_table(df_paths, metric_cols_paths)
    disp_paths.insert(0, "run", df_paths["run"].iloc[0] if "run" in df_paths.columns and len(df_paths) else run_dir.name)
    disp_paths.to_csv(rep / "dispersion_over_paths.csv", index=False)

    # --- 2) Dispersion über alle Years (alle Path-Year Zeilen zusammen) ---
    exclude_years = {"run", "path_id", "year"}
    metric_cols_years = [c for c in df_years.columns if c not in exclude_years]
    metric_cols_years = [c for c in metric_cols_years if pd.api.types.is_numeric_dtype(df_years[c])]

    disp_years_all = _stats_table(df_years, metric_cols_years)
    disp_years_all.insert(0, "run", df_years["run"].iloc[0] if "run" in df_years.columns and len(df_years) else run_dir.name)
    disp_years_all.to_csv(rep / "dispersion_over_years_all.csv", index=False)

    # --- 3) Dispersion pro Path über Years ---
    rows = []
    for pid, g in df_years.groupby("path_id", sort=True):
        tab = _stats_table(g, metric_cols_years)
        tab.insert(0, "path_id", int(pid))
        rows.append(tab)

    disp_years_by_path = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not disp_years_by_path.empty:
        disp_years_by_path.insert(
            0, "run",
            df_years["run"].iloc[0] if "run" in df_years.columns and len(df_years) else run_dir.name
        )
        disp_years_by_path.to_csv(rep / "dispersion_over_years_by_path.csv", index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="z.B. .../SB3_Defaults")
    args = ap.parse_args()

    main(args.run_dir)
