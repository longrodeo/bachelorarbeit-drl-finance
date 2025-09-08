# -*- coding: utf-8 -*-
"""
End-to-End Smoke mit Accounting + Evaluator
- Läuft die TradingEnv (10 Schritte, deterministisch)
- Schreibt Parquets via AccountingRecorder
- Baut anschließend Rewards/NAV via Evaluator
- Erzeugt eine kleine Summary-CSV

Voraussetzung: eure Config-/Pfad-Helfer & Module sind im PYTHONPATH (Projektstruktur).
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
import numpy as np

# --- Pfade/Helper laden -------------------------------------------------------
from src.utils.paths import (
    CLEAN_PANEL, RISKFREE_NORM_FILE, SPEC_S0_YAML, ACCOUNT_DIR,
    get_asset_groups, get_assets_flat,
)
from src.utils.parquet_io import load_parquet, save_parquet
from src.env.smoke_test_env import main as smoke_main

# Recorder & Evaluator
import src.accounting.recorder as recoder  # AccountingRecorder
from src.accounting.evaluator import compute_rewards_from_snapshots
from src.accounting.reward import RewardSpec

# State-Building & Portfolio
from src.state.state_builder import load_spec, build_state_for_date
from src.portfolio.broker import PortfolioLite


def run_env_and_record(account_dir: Path, steps: int = 10) -> None:
    """
    Lädt Panel & Riskfree, baut Env mit AccountingRecorder und
    fährt den deterministischen Smoke (Cash ↔ Equal).
    """
    # 1) Daten laden
    panel_clean = load_parquet(CLEAN_PANEL)
    riskfree    = load_parquet(RISKFREE_NORM_FILE)

    # Handelstage & Assets (stabile Reihenfolge aus assets.yml / SPEC)
    dates  = panel_clean.index.get_level_values(0).unique().sort_values()
    assets = get_assets_flat(get_asset_groups())

    # Riskfree: daily factor (z. B. "daily_factor_360")
    # → exakt wie im bestehenden Smoke-Runner genutzt
    rf_factor = riskfree["daily_factor_360"].reindex(dates).to_numpy()

    # 2) State-Spec + Builder
    spec = load_spec(SPEC_S0_YAML)
    state_builder = SimpleNamespace(build_state_for_date=build_state_for_date)

    # 3) Portfolio & Recorder (Accounting → Parquets)
    portfolio = PortfolioLite(assets=assets, initial_cash=1_000_000.0)
    recorder  = recoder.AccountingRecorder(out_dir=account_dir)

    # 4) INJECT & Run
    INJECT = {
        "panel_clean": panel_clean,
        "dates": dates,
        "assets": assets,
        "spec": spec,
        "state_builder": state_builder,
        "portfolio": portfolio,
        "rf_factor": rf_factor,
        "recorder": recorder,
        # Reward/Evaluator bei Bedarf:
        "reward_kind": "log",
        "start_idx": 0,
        "end_idx_exclusive": len(dates),
    }

    # Der Smoke macht intern standardmäßig ~10 Schritte; falls du mehr/weniger willst,
    # könntest du in smoke_test_env.py MAX_STEPS anpassen. Hier rufen wir ihn direkt auf.
    smoke_main(INJECT)


def evaluate_accounting_and_write_summary(account_dir: Path) -> Path:
    """
    Evaluator über die geschriebenen Snapshots laufen lassen
    und eine kleine Zusammenfassung als CSV schreiben.
    """
    snaps_path = account_dir / "portfolio_snapshots.parquet"
    if not snaps_path.exists():
        raise FileNotFoundError(f"Snapshots nicht gefunden: {snaps_path}")

    # 1) Rewards/NAV via Evaluator bauen (speichert auch Parquet)
    #    out_name passend zu eurer Pfad-Konvention setzen:
    rewards_df = compute_rewards_from_snapshots(
        accounting_dir=account_dir,
        spec=RewardSpec(kind="icvar_dd", alpha=0.05, icvar_mode="ex_post", ewm_alpha=None),
        out_name="rewards_log.parquet",
    )

    # 2) Kleine Summary (NAV, Return, Vol, MDD)
    #    rewards_df enthält mind.: round, t, nav_t, nav_t-1, r_log_t, reward_t (je nach Spec)
    df = rewards_df.sort_values("round").reset_index(drop=True)
    nav = df["nav_t"].astype(float).ffill()

    # Kenngrößen
    nav0   = float(nav.iloc[1]) if len(nav) > 1 else float(nav.iloc[0])
    nav_T  = float(nav.iloc[-1])
    total_return = (nav_T / max(1e-12, nav0)) - 1.0

    r_log = df["r_log_t"].astype(float)
    mu    = float(r_log.mean(skipna=True))
    sigma = float(r_log.std(skipna=True))
    ann_mu    = mu * 252.0
    ann_sigma = sigma * np.sqrt(252.0)
    # MDD über NAV_Pfad
    peak = nav.cummax()
    mdd  = ((peak - nav) / peak.replace(0, np.nan)).fillna(0.0)
    mdd_max = float(mdd.max())

    # 3) CSV schreiben
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = account_dir / f"summary_e2e_{ts}.csv"
    summary = pd.DataFrame([{
        "start": str(df["t"].iloc[0]) if len(df) else "",
        "end":   str(df["t"].iloc[-1]) if len(df) else "",
        "rounds": int(len(df)),
        "nav_start": nav0,
        "nav_end": nav_T,
        "total_return": total_return,
        "mean_r_log": mu,
        "std_r_log": sigma,
        "ann_mean_r_log": ann_mu,
        "ann_std_r_log": ann_sigma,
        "max_drawdown": mdd_max,
    }])
    save_parquet(rewards_df, account_dir / "rewards_log_copy.parquet")  # optional: Kopie
    summary.to_csv(out, index=False)

    print(f"[EVAL] Rewards -> {account_dir / 'rewards_log.parquet'}")
    print(f"[EVAL] Summary CSV -> {out}")
    print(summary.to_string(index=False))
    return out


def main():
    account_dir = Path(ACCOUNT_DIR)
    account_dir.mkdir(parents=True, exist_ok=True)

    print("[RUN] Env + Accounting…")
    run_env_and_record(account_dir)

    print("[RUN] Evaluator…")
    _ = evaluate_accounting_and_write_summary(account_dir)

    print("[DONE] E2E-Smoke vollständig.")

if __name__ == "__main__":
    main()
