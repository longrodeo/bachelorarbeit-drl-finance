from pathlib import Path
import pandas as pd
import numpy as np

# Wir nutzen eure bestehenden KPI-Funktionen
from src.eval.metrics import scorecard_baseline, scorecard_train, psr
from src.utils.parquet_io import load_parquet

# ------------------------------------------------------------
# Liest Path_XX/portfolio_snapshots.parquet (+ optional trade_events, rewards),
# berechnet:
# - NAV (liegt in portfolio_snapshots als 'portfolio_value_t')
# - Returns: pct_change(NAV)
# - cost_rate: fees_total_round / NAV_{t-1} (aus snapshots)
# - turnover: sum(notional_abs)/NAV_{t-1} (aus trade_events)
# und schreibt _report/metrics_per_path.csv mit PSR, DSR, Sharpe, CumReturn, MaxDD, etc.
# ------------------------------------------------------------

def _load_path(path_dir: Path):
    snaps_f = path_dir / "portfolio_snapshots.parquet"
    if not snaps_f.is_file():
        return None
    snaps = load_parquet(snaps_f).sort_values("t")
    snaps["t"] = pd.to_datetime(snaps["t"])
    # NAV & Returns
    pv = snaps.drop_duplicates("t").set_index("t")["portfolio_value_t"].astype(float)
    ret_total = pv.pct_change().dropna()

    # Kosten (aus snapshots)
    fees_t = snaps.groupby("t")["fees_total_round"].first().astype(float)
    prev_pv = pv.shift(1)
    cost_rate = (fees_t / prev_pv).reindex(ret_total.index).fillna(0.0)

    # Turnover (aus trade_events, optional)
    ev_f = path_dir / "trade_events.parquet"
    if ev_f.is_file():
        ev = load_parquet(ev_f)
        if "t" in ev.columns:
            ev["t"] = pd.to_datetime(ev["t"])
        notional_t = ev.groupby("t")["notional_abs"].sum().astype(float)
        turnover = (notional_t / prev_pv).reindex(ret_total.index).fillna(0.0)
    else:
        turnover = pd.Series(0.0, index=ret_total.index, name="turnover")

    return ret_total, cost_rate, turnover

def compute_path_metrics(run_dir: str | Path,
                         periods_per_year: int = 252,
                         n_trials_dsr: int = 48,
                         riskfree_parquet: str | None = None) -> None:
    run_dir = Path(run_dir)
    rep = run_dir / "_report"
    rep.mkdir(parents=True, exist_ok=True)

    # optional Risk-Free: falls vorhanden, Nutzen für Excess-Returns
    rf = None
    if riskfree_parquet:
        rfp = Path(riskfree_parquet)
        if rfp.is_file():
            rf_df = load_parquet(rfp).squeeze()
            rf = pd.Series(rf_df.values, index=pd.to_datetime(rf_df.index), dtype=float)

    rows = []
    for p in range(1, 6):
        pdir = run_dir / f"Path_{p:02d}"
        if not (pdir / "portfolio_snapshots.parquet").is_file():
            continue

        loaded = _load_path(pdir)
        if loaded is None:
            continue
        ret_total, cost_rate, turnover = loaded

        # Excess-Returns (optional)
        if rf is not None:
            rf_a = rf.reindex(ret_total.index).fillna(0.0)
            ret_ex = (1.0 + ret_total).div(1.0 + rf_a) - 1.0
        else:
            ret_ex = ret_total

        # KPIs (eure Funktionen)
        base_total = scorecard_baseline(ret_total, periods_per_year=periods_per_year,
                                        cost_rate=cost_rate, turnover=turnover)
        base_ex    = scorecard_baseline(ret_ex,    periods_per_year=periods_per_year,
                                        cost_rate=cost_rate, turnover=turnover)
        train_ex   = scorecard_train(ret_ex, n_trials_for_dsr=n_trials_dsr,
                                     cost_rate=cost_rate, turnover=turnover)
        try:
            psr_ex = psr(ret_ex, sr_threshold=0.0)
        except Exception:
            psr_ex = np.nan

        rows.append({
            "run": run_dir.name,
            "path_id": p,
            # wichtige KPIs kompakt
            "total_cum_return": base_total.get("cum_return", np.nan),
            "ex_cum_return":    base_ex.get("cum_return", np.nan),
            "ex_sharpe":        base_ex.get("sharpe", np.nan),
            "ex_vol_ann":       base_ex.get("vol_ann", np.nan),
            "total_maxdd":      base_total.get("maxdd", np.nan),
            "total_total_cost": base_total.get("total_cost", np.nan),
            "ex_roec":          base_ex.get("roec", np.nan),
            "psr_ex":           psr_ex,
            "dsr_ex":           train_ex.get("dsr", np.nan),
        })

    if rows:
        pd.DataFrame(rows).sort_values("path_id").to_csv(rep / "metrics_per_path.csv", index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--periods_per_year", type=int, default=252)
    ap.add_argument("--n_trials_dsr", type=int, default=48)
    ap.add_argument("--riskfree_parquet", type=str, default=None)
    args = ap.parse_args()
    compute_path_metrics(args.run_dir,
                         periods_per_year=args.periods_per_year,
                         n_trials_dsr=args.n_trials_dsr,
                         riskfree_parquet=args.riskfree_parquet)
