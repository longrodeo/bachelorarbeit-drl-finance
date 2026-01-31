from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


def _to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def _stats_table(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    x = _to_numeric_frame(df, metric_cols)

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


def _rep_dir(run_dir: Path) -> Path:
    # erlaubt: run_dir = .../E2 oder .../benchmark_fin_per_year oder direkt .../_report
    return run_dir if run_dir.name == "_report" else (run_dir / "_report")


def _infer_run_label(run_dir: Path, df: pd.DataFrame) -> str:
    if "run" in df.columns and len(df):
        v = str(df["run"].iloc[0])
        if v and v.lower() != "nan":
            return v

    # schöner Default, falls run_dir=.../E2
    if run_dir.name.lower() == "e2" and run_dir.parent is not None and run_dir.parent.parent is not None:
        # .../<config>/<run_id>/E2
        return f"{run_dir.parent.parent.name}/{run_dir.parent.name}"

    return run_dir.name


def _select_numeric_metric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = [c for c in df.columns if c not in exclude]
    numeric_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            numeric_cols.append(c)
    return numeric_cols


def _filter_years(df: pd.DataFrame, years: list[int] | None) -> pd.DataFrame:
    if years is None or len(years) == 0:
        return df

    year_col = None
    for cand in ["year", "test_year", "testyear"]:
        if cand in df.columns:
            year_col = cand
            break

    if year_col is None:
        # fallback: suche irgendeine Spalte mit "year"
        for c in df.columns:
            if "year" in c.lower():
                year_col = c
                break

    if year_col is None:
        raise ValueError(
            "Konnte keine Year-Spalte finden (erwartet z.B. 'year' oder 'test_year'). "
            f"Spalten: {list(df.columns)}"
        )

    y = df[year_col].astype(str).str.extract(r"(\d{4})")[0]
    y = pd.to_numeric(y, errors="coerce")

    out = df.copy()
    out["_year"] = y
    out = out[out["_year"].isin(years)]
    out = out.drop(columns=["_year"])
    return out


def _mode_auto(rep: Path) -> str:
    if (rep / "metrics_per_test_year.csv").is_file():
        return "wf"
    if (rep / "metrics_per_path.csv").is_file() and (rep / "metrics_per_path_year.csv").is_file():
        return "cpcv"
    raise FileNotFoundError(
        "Konnte keinen Modus erkennen. Erwartet entweder:\n"
        "- WF: _report/metrics_per_test_year.csv\n"
        "- CPCV: _report/metrics_per_path.csv und _report/metrics_per_path_year.csv\n"
        f"Gefunden in: {rep}"
    )


def run_cpcv(run_dir: Path) -> None:
    rep = _rep_dir(run_dir)

    f_paths = rep / "metrics_per_path.csv"
    f_years = rep / "metrics_per_path_year.csv"

    if not f_paths.is_file():
        raise FileNotFoundError(f"Fehlt: {f_paths}")
    if not f_years.is_file():
        raise FileNotFoundError(f"Fehlt: {f_years}")

    df_paths = pd.read_csv(f_paths)
    df_years = pd.read_csv(f_years)

    # --- 1) Dispersion über Paths (1..N) ---
    exclude_paths = {"run", "path_id"}
    metric_cols_paths = _select_numeric_metric_cols(df_paths, exclude_paths)

    disp_paths = _stats_table(df_paths, metric_cols_paths)
    disp_paths.insert(0, "run", _infer_run_label(run_dir, df_paths))
    disp_paths.to_csv(rep / "dispersion_over_paths.csv", index=False)

    # --- 2) Dispersion über alle Years (alle Path-Year Zeilen zusammen) ---
    exclude_years = {"run", "path_id", "year"}
    metric_cols_years = _select_numeric_metric_cols(df_years, exclude_years)

    disp_years_all = _stats_table(df_years, metric_cols_years)
    disp_years_all.insert(0, "run", _infer_run_label(run_dir, df_years))
    disp_years_all.to_csv(rep / "dispersion_over_years_all.csv", index=False)

    # --- 3) Dispersion pro Path über Years ---
    rows = []
    for pid, g in df_years.groupby("path_id", sort=True):
        tab = _stats_table(g, metric_cols_years)
        tab.insert(0, "path_id", int(pid))
        rows.append(tab)

    disp_years_by_path = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not disp_years_by_path.empty:
        disp_years_by_path.insert(0, "run", _infer_run_label(run_dir, df_years))
        disp_years_by_path.to_csv(rep / "dispersion_over_years_by_path.csv", index=False)


def run_wf(run_dir: Path, years: list[int] | None = None) -> None:
    rep = _rep_dir(run_dir)
    f = rep / "metrics_per_test_year.csv"
    if not f.is_file():
        raise FileNotFoundError(f"Fehlt: {f}")

    df = pd.read_csv(f)
    df = _filter_years(df, years)

    exclude = {
        "run",
        "year", "test_year", "testyear",
        "fold", "fold_id",
        "window", "train_start", "train_end", "test_start", "test_end",
    }
    metric_cols = _select_numeric_metric_cols(df, exclude)
    if not metric_cols:
        raise ValueError(f"Keine numerischen KPI-Spalten gefunden in {f}. Spalten: {list(df.columns)}")

    disp = _stats_table(df, metric_cols)
    disp.insert(0, "run", _infer_run_label(run_dir, df))

    if years and len(years) > 0:
        disp.insert(1, "year_from", min(years))
        disp.insert(2, "year_to", max(years))
        disp.insert(3, "n_years", len(years))

    disp.to_csv(rep / "dispersion_over_test_years.csv", index=False)


def main(run_dir: str | Path, mode: str = "auto", years: list[int] | None = None) -> None:
    run_dir = Path(run_dir)
    rep = _rep_dir(run_dir)

    if mode == "auto":
        mode = _mode_auto(rep)

    if mode == "cpcv":
        run_cpcv(run_dir)
    elif mode == "wf":
        run_wf(run_dir, years=years)
    else:
        raise ValueError("mode muss 'auto', 'cpcv' oder 'wf' sein")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="z.B. .../E2 oder .../benchmark_fin_per_year oder .../_report")
    ap.add_argument("--mode", default="auto", choices=["auto", "cpcv", "wf"])
    ap.add_argument("--years", type=int, nargs="*", default=None, help="z.B. 2018 2019 ... 2024 (optional)")
    args = ap.parse_args()

    main(args.run_dir, mode=args.mode, years=args.years)
