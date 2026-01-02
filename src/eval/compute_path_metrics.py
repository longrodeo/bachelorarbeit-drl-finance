from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.parquet_io import load_parquet  # zentraler Loader mit Engine-Fallbacks
from src.eval.metrics import simple_from_log, scorecard_eval, psr, dsr


def _load_returns_from_rewards(path_dir: Path, log_col: str = "r_log_t") -> pd.Series:
    f = path_dir / "rewards.parquet"
    df = load_parquet(f)

    if "t" not in df.columns:
        raise ValueError(f"'t' fehlt in {f}")
    if log_col not in df.columns:
        raise ValueError(f"'{log_col}' fehlt in {f}")

    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values("t").drop_duplicates("t", keep="last").set_index("t")

    r_log = df[log_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    r = simple_from_log(r_log)
    r.name = "ret_total"
    return r


def _load_cost_turnover(path_dir: Path) -> tuple[float, float]:
    # optionales Reporting; falls Dateien/Spalten fehlen -> NaN
    ps_f = path_dir / "portfolio_snapshots.parquet"
    if not ps_f.is_file():
        return (float("nan"), float("nan"))

    ps = load_parquet(ps_f)
    need = {"t", "portfolio_value_t", "fees_total_round"}
    if not need.issubset(ps.columns):
        return (float("nan"), float("nan"))

    ps["t"] = pd.to_datetime(ps["t"], utc=True)
    ps = ps.sort_values("t").set_index("t")

    pv = ps["portfolio_value_t"].astype(float)
    fees = ps["fees_total_round"].astype(float).fillna(0.0)

    prev_pv = pv.shift(1)
    cost_rate = (fees / prev_pv).replace([np.inf, -np.inf], np.nan).dropna()
    avg_cost = float(cost_rate.mean()) if len(cost_rate) else float("nan")

    ev_f = path_dir / "trade_events.parquet"
    if not ev_f.is_file():
        return (avg_cost, float("nan"))

    ev = load_parquet(ev_f)
    if "t" not in ev.columns or "notional_abs" not in ev.columns:
        return (avg_cost, float("nan"))

    ev["t"] = pd.to_datetime(ev["t"], utc=True)
    notional_t = ev.groupby("t")["notional_abs"].sum().astype(float)

    turnover = (notional_t / prev_pv).replace([np.inf, -np.inf], np.nan).dropna()
    avg_to = float(turnover.mean()) if len(turnover) else float("nan")

    return (avg_cost, avg_to)


def _load_riskfree_series(riskfree_parquet: str | None, riskfree_col: str = "rf_daily_rate") -> pd.Series | None:
    if not riskfree_parquet:
        return None
    rfp = Path(riskfree_parquet)
    if not rfp.is_file():
        return None

    rf_df = load_parquet(rfp)

    # Fall A: 't' als Spalte vorhanden
    if isinstance(rf_df, pd.DataFrame) and "t" in rf_df.columns:
        rf_df["t"] = pd.to_datetime(rf_df["t"], utc=True)
        if riskfree_col not in rf_df.columns:
            raise ValueError(f"riskfree_col='{riskfree_col}' nicht gefunden. Spalten: {list(rf_df.columns)}")
        s = rf_df.sort_values("t").set_index("t")[riskfree_col].astype(float)
    else:
        # Fall B: Datetime-Index, mehrere Spalten
        if isinstance(rf_df, pd.DataFrame):
            if riskfree_col in rf_df.columns:
                s = rf_df[riskfree_col].astype(float)
            elif rf_df.shape[1] == 1:
                s = rf_df.iloc[:, 0].astype(float)
            else:
                raise ValueError(f"riskfree parquet hat mehrere Spalten {list(rf_df.columns)} – "
                                 f"bitte --riskfree_col angeben.")


        s.index = pd.to_datetime(s.index, utc=True)

    s.name = "rf_daily"
    return s


def compute_path_metrics(
    run_dir: str | Path,
    *,
    periods_per_year: int = 252,
    alpha_cvar: float = 0.95,
    riskfree_parquet: str | None = None,
    riskfree_col: str,
    include_psr: bool = False,
    include_dsr: bool = False,
    n_trials_dsr: int = 1,
    sr_var_trials: float | None = None,
) -> None:
    run_dir = Path(run_dir)
    rep = run_dir / "_report"
    rep.mkdir(parents=True, exist_ok=True)

    rows_path = []
    rows_year = []
    rf = _load_riskfree_series(riskfree_parquet, riskfree_col=riskfree_col)

    for pid in range(1, 6):
        pdir = run_dir / f"Path_{pid:02d}"

        # returns aus rewards.parquet (r_log_t -> simple returns)
        ret_total = _load_returns_from_rewards(pdir, log_col="r_log_t")

        # excess returns (optional)
        if rf is not None:
            rf_a = rf.reindex(ret_total.index).fillna(0.0)
            ret_ex = (1.0 + ret_total).div(1.0 + rf_a) - 1.0
        else:
            ret_ex = ret_total.copy()
        ret_ex.name = "ret_ex"

        # pro Jahr (excess)
        for y, r_y in ret_ex.groupby(ret_ex.index.year):
            k = scorecard_eval(r_y, periods_per_year=periods_per_year, alpha_cvar=alpha_cvar)
            row_y = {
                "run": run_dir.name,
                "path_id": pid,
                "year": int(y),
                "ex_cum_return": k["cum_return"],
                "ex_sharpe": k["sharpe"],
                "ex_sortino": k["sortino"],
                "ex_cvar_95": k["cvar"],
                "ex_maxdd": k["maxdd"],
                "ex_calmar": k["calmar"],
            }
            if include_psr:
                row_y["psr_ex"] = float(psr(r_y, sr_threshold=0.0))
            else:
                row_y["psr_ex"] = float("nan")
            rows_year.append(row_y)

        # aggregiert pro Path (total/excess)
        k_total = scorecard_eval(ret_total, periods_per_year=periods_per_year, alpha_cvar=alpha_cvar)
        k_ex = scorecard_eval(ret_ex, periods_per_year=periods_per_year, alpha_cvar=alpha_cvar)

        avg_cost, avg_to = _load_cost_turnover(pdir)

        row = {
            "run": run_dir.name,
            "path_id": pid,

            "total_cum_return": k_total["cum_return"],
            "total_maxdd": k_total["maxdd"],

            "ex_cum_return": k_ex["cum_return"],
            "ex_sharpe": k_ex["sharpe"],
            "ex_sortino": k_ex["sortino"],
            "ex_cvar_95": k_ex["cvar"],
            "ex_calmar": k_ex["calmar"],

            "avg_cost_rate": avg_cost,
            "avg_turnover": avg_to,
        }

        if include_psr:
            row["psr_ex"] = float(psr(ret_ex, sr_threshold=0.0))
        else:
            row["psr_ex"] = float("nan")

        if include_dsr:
            row["dsr_ex"] = float(dsr(ret_ex, n_trials=n_trials_dsr, sr_var_trials=sr_var_trials))
        else:
            row["dsr_ex"] = float("nan")

        rows_path.append(row)

    if rows_year:
        pd.DataFrame(rows_year).sort_values(["path_id", "year"]).to_csv(rep / "metrics_per_path_year.csv", index=False)

    if rows_path:
        pd.DataFrame(rows_path).sort_values("path_id").to_csv(rep / "metrics_per_path.csv", index=False)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--periods_per_year", type=int, default=252)
    ap.add_argument("--alpha_cvar", type=float, default=0.95)
    ap.add_argument("--riskfree_parquet", type=str, default=None)
    ap.add_argument("--riskfree_col", type=str, default="rf_daily_rate",
                    help="Spalte im riskfree.parquet (z.B. rf_daily_rate oder rf_daily_factor)")

    ap.add_argument("--include_psr", action="store_true")
    ap.add_argument("--include_dsr", action="store_true")
    ap.add_argument("--n_trials_dsr", type=int, default=1)
    ap.add_argument("--sr_var_trials", type=float, default=None)

    args = ap.parse_args()

    compute_path_metrics(
        args.run_dir,
        periods_per_year=args.periods_per_year,
        alpha_cvar=args.alpha_cvar,
        riskfree_parquet=args.riskfree_parquet,
        riskfree_col=args.riskfree_col,
        include_psr=args.include_psr,
        include_dsr=args.include_dsr,
        n_trials_dsr=args.n_trials_dsr,
        sr_var_trials=args.sr_var_trials,
    )
