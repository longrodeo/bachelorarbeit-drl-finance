from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# Defaults
WF_ROOT = Path(r"C:\Dev\Bachelorarbeit\results\accounting\runs\wf_final_2")
BENCH_REPORT = Path(r"C:\Dev\Bachelorarbeit\data\benchmark_fin_per_year\_report\dispersion_over_test_years.csv")
OUT_DIR = WF_ROOT / "_plots_wf" / "summary"


def newest_dispersion_per_config(wf_root: Path) -> dict[str, Path]:
    best: dict[str, tuple[str, Path]] = {}
    for p in wf_root.rglob(r"_report/dispersion_over_test_years.csv"):
        rel = p.relative_to(wf_root).parts
        if len(rel) < 4:
            continue
        cfg, run_id = rel[0], rel[1]
        if str(cfg).startswith("_"):
            continue
        m = re.search(r"(\d{8}_\d{6})", run_id)
        ts = m.group(1) if m else run_id
        if (cfg not in best) or (ts > best[cfg][0]):
            best[cfg] = (ts, p)
    return {cfg: path for cfg, (_, path) in best.items()}


def latex_config_label(cfg: str) -> str:
    # Reward
    if cfg.startswith("log_"):
        reward = r"\Rlog"
    elif cfg.startswith("icvar_dd_"):
        reward = r"\RlogICVaRDeltaMDD"
    elif cfg.startswith("icvar_"):
        reward = r"\RlogICVaR"
    else:
        reward = cfg.replace("_", r"\_")

    # S0 / S1 (explizit im Label, mit Leerzeichen)
    mS = re.search(r"_S([01])", cfg)
    s = f"S{mS.group(1)}" if mS else ""

    return rf"{reward} {s}".strip()


def col_label(metric: str) -> str:
    # ohne Unterstriche, paper-tauglich
    mapping = {
        "ex_cum_return":  r"Excess Return [\%] $\uparrow$",
        "total_maxdd":    r"Max Drawdown [\%] $\downarrow$",
        "ex_cvar_95":     r"Excess CVaR$_{95}$ [\%] $\uparrow$",
        "ex_sortino":     r"Excess Sortino $\uparrow$",
        "avg_turnover":   r"Avg. Turnover [\%] $\downarrow$",
        "avg_cost_rate":  r"Avg. Cost Rate [bp] $\downarrow$",
    }
    return mapping.get(metric, metric.replace("_", " "))


def format_cell(metric: str, median: float, q25: float, q75: float) -> str:
    if pd.isna(median) or pd.isna(q25) or pd.isna(q75):
        return ""

    # Returns: decimal -> %
    if metric == "ex_cum_return":
        median, q25, q75 = 100 * median, 100 * q25, 100 * q75
        return f"{median:.2f} [{q25:.2f}; {q75:.2f}]"

    # MDD: bei dir negativ gespeichert -> als positive Magnitude in %
    # Achtung: Quartile drehen sich beim Vorzeichenflip!
    if metric == "total_maxdd":
        median_mag = -median
        q25_mag = -q75
        q75_mag = -q25
        median_mag, q25_mag, q75_mag = 100 * median_mag, 100 * q25_mag, 100 * q75_mag
        return f"{median_mag:.2f} [{q25_mag:.2f}; {q75_mag:.2f}]"

    # CVaR: meist negativ, aber "höher (= weniger negativ) ist besser" -> in % anzeigen
    if metric == "ex_cvar_95":
        median, q25, q75 = 100 * median, 100 * q25, 100 * q75
        return f"{median:.2f} [{q25:.2f}; {q75:.2f}]"

    # Ratios
    if metric == "ex_sortino":
        return f"{median:.2f} [{q25:.2f}; {q75:.2f}]"

    # Turnover: decimal -> %
    if metric == "avg_turnover":
        median, q25, q75 = 100 * median, 100 * q25, 100 * q75
        return f"{median:.1f} [{q25:.1f}; {q75:.1f}]"

    # Cost rate: decimal -> bp
    if metric == "avg_cost_rate":
        median, q25, q75 = 10000 * median, 10000 * q25, 10000 * q75
        return f"{median:.2f} [{q25:.2f}; {q75:.2f}]"

    return f"{median:.4f} [{q25:.4f}; {q75:.4f}]"


def build_table(wf_root: Path, bench_report: Path, out_dir: Path, years: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hauptkapitel: alles außer total return, ex calmar, ex sharpe
    metrics = [
        "ex_cum_return",
        "total_maxdd",
        "ex_cvar_95",
        "ex_sortino",
        "avg_turnover",
        "avg_cost_rate",
    ]

    rows: dict[str, dict[str, str]] = {}

    # Benchmark
    bdf = pd.read_csv(bench_report).set_index("metric")
    rows[r"Benchmark (B\&H 85/10/5)"] = {
        col_label(m): (
            format_cell(m, bdf.loc[m, "median"], bdf.loc[m, "q25"], bdf.loc[m, "q75"])
            if m in bdf.index else ""
        )
        for m in metrics
    }

    # Configs
    cfg_paths = newest_dispersion_per_config(wf_root)
    if not cfg_paths:
        raise FileNotFoundError(f"Keine dispersion_over_test_years.csv unter {wf_root} gefunden.")

    for cfg, p in sorted(cfg_paths.items()):
        df = pd.read_csv(p).set_index("metric")
        label = latex_config_label(cfg)
        rows[label] = {
            col_label(m): (
                format_cell(m, df.loc[m, "median"], df.loc[m, "q25"], df.loc[m, "q75"])
                if m in df.index else ""
            )
            for m in metrics
        }

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Konfiguration"

    csv_path = out_dir / "wf_kpi_table_median_q25_q75.csv"
    tex_path = out_dir / "wf_kpi_table_median_q25_q75.tex"
    table.to_csv(csv_path)

    years_txt = f"{min(years)}--{max(years)}" if years else "2018--2024"

    latex_body = table.to_latex(
        escape=False,
        index=True,
        column_format="l" + "r" * len(table.columns),
    )

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + latex_body +
        "}\n"
        f"\\caption{{Walk-Forward OOS KPI-Summary über Testjahre {years_txt}. "
        "Zellen zeigen den Median; in eckigen Klammern das 25\\%- und 75\\%-Quartil (Q25; Q75). "
        "Hinweis: Bei CVaR gilt \\emph{höher} (= weniger negativ) ist besser.}}"
        "\\label{tab:wf_kpi_summary}\n"
        "\\end{table}\n"
    )
    tex_path.write_text(tex, encoding="utf-8")

    print("Saved:")
    print("-", csv_path)
    print("-", tex_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wf_root", default=str(WF_ROOT))
    ap.add_argument("--bench_report", default=str(BENCH_REPORT))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--years", type=int, nargs="*", default=[2018, 2019, 2020, 2021, 2022, 2023, 2024])
    args = ap.parse_args()

    build_table(
        wf_root=Path(args.wf_root),
        bench_report=Path(args.bench_report),
        out_dir=Path(args.out_dir),
        years=args.years,
    )
