# src/eval/evaluate_logs.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any
from src.eval.metrics import scorecard_train, scorecard_baseline  # aus deinem kpis.py

def evaluate_run(
    out_dir: Path | str,
    rf: Optional[pd.Series] = None,          # täglicher rf, Index= t
    n_trials_for_dsr: int = 48,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    snaps = pd.read_parquet(out_dir / "portfolio_snapshots.parquet").copy()
    snaps = snaps.sort_values("t")

    # ---- 1) Zeitreihen (ein Punkt je t) ----
    pv = snaps.drop_duplicates("t").set_index("t")["portfolio_value_t"].astype(float)
    ret_total = pv.pct_change().dropna()

    fees_t = snaps.groupby("t")["fees_total_round"].first().astype(float)
    prev_pv = pv.shift(1)
    cost_rate = (fees_t / prev_pv).reindex(ret_total.index).fillna(0.0)

    # Turnover aus Events (optional)
    evt_path = out_dir / "trade_events.parquet"
    turnover = None
    if evt_path.exists():
        evts = pd.read_parquet(evt_path)
        notional_t = evts.groupby("t")["notional_abs"].sum().astype(float)
        turnover = (notional_t / prev_pv).reindex(ret_total.index).fillna(0.0)

    # ---- 2) Excess-Returns (optional, sonst total) ----
    if rf is not None:
        rf = rf.reindex(ret_total.index).astype(float)
        ret_ex = (1.0 + ret_total).div(1.0 + rf) - 1.0
    else:
        ret_ex = ret_total

    # ---- 3) Scorecards ----
    train_ex = scorecard_train(
        ret_ex, n_trials_for_dsr=n_trials_for_dsr,
        cost_rate=cost_rate, turnover=turnover, fee_bps=0.0, slippage_bps=0.0,
    )
    base_ex = scorecard_baseline(
        ret_ex, periods_per_year=periods_per_year,
        cost_rate=cost_rate, turnover=turnover, fee_bps=0.0, slippage_bps=0.0,
    )
    # Leserfreundlich zusätzlich eine Total-Variante
    base_total = scorecard_baseline(
        ret_total, periods_per_year=periods_per_year,
        cost_rate=cost_rate, turnover=turnover, fee_bps=0.0, slippage_bps=0.0,
    )

    return {
        "train_excess": train_ex,        # DSR, DD95, TuW95, RoEC, CumReturn (excess)
        "baseline_excess": base_ex,      # CumReturn, Sharpe, Vol, MaxDD, TotalCost, RoEC (excess)
        "baseline_total": base_total,    # dasselbe auf Total-Returns
    }
