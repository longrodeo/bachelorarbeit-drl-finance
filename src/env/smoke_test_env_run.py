# src/run_smoke_env.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from src.utils.paths import FEATURES_NORM, RISKFREE_NORM_FILE, SPEC_S0_YAML, get_asset_groups, get_assets_flat, CLEAN_PANEL
from src.utils.parquet_io import load_parquet
from src.env.smoke_test_env import main

# --- 1) Laden über eure Helper ---
panel_clean = load_parquet(CLEAN_PANEL)
panel_features = load_parquet(FEATURES_NORM)
riskfree    = load_parquet(RISKFREE_NORM_FILE)

# Handelstage & Assets (stabile Reihenfolge aus der Config)
dates  = panel_clean.index.get_level_values(0).unique().sort_values()
assets = get_assets_flat(get_asset_groups())

# Cash-Faktor stumpf aus daily_factor (keine ifs)
rf_factor = riskfree["daily_factor_360"].reindex(dates).to_numpy()
rf_rate = riskfree["risk_free_annual_z"].reindex(dates).to_numpy()

# --- 2) State-Spec laden & Builder „einpacken“ ---
from src.state.state_builder import load_spec, build_state_for_date
spec = load_spec(SPEC_S0_YAML)
state_builder = SimpleNamespace(build_state_for_date=build_state_for_date)  # hat die geforderte Methode

# --- 3) Portfolio-Instanz (kein .step übergeben!) ---
from src.portfolio.broker import PortfolioLite
portfolio = PortfolioLite(assets=assets, initial_cash=1_000_000.0)  # fee_kwargs etc. bei Bedarf hier setzen

# --- 4) Minimaler Recorder (in-memory) ---
class ListRecorder:
    def __init__(self): self.rows = []
    def log_round(self, t, assets, p1, cash, shares, w_post, exec_df, fees_df, round_id):
        self.rows.append({
            "date": t,
            "round_id": int(round_id),
            "cash": float(cash),
            "n_assets_px": int(getattr(p1, "size", len(p1))),  # Anzahl Preise, nicht float(p1)!
            "n_prices": int(getattr(p1, "size", len(p1))),
            "w_post": getattr(w_post, "tolist", lambda: list(w_post))(),
        })

recorder = ListRecorder()

# --- 5) INJECT bauen und Smoke-Test fahren ---
INJECT = {
    "panel_clean": panel_clean,
    "panel_features": panel_features,
    "dates": dates,
    "assets": assets,
    "spec": spec,
    "state_builder": state_builder,   # Objekt mit build_state_for_date(...)
    "portfolio": portfolio,           # PortfolioLite-Instanz (hat reset/step)
    "rf_factor": rf_factor,           # roher daily_factor, gleich lang wie dates-Fenster
    "rf_rate": rf_rate,
    "recorder": recorder,
    # "evaluator": evaluator, "reward_scaler": reward_scaler,
    "reward_kind": "log", "start_idx": 0, "end_idx_exclusive": len(dates),
}

main(INJECT)  # -> reset + 10 deterministische Schritte + Prints

# --- 6) Recorder-Log speichern (einfach) ---
outdir = Path("logs"); outdir.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out = outdir / f"smoke_env_s0_{ts}.csv"
pd.DataFrame(recorder.rows).to_csv(out, index=False)
print(f"[LOG] {len(recorder.rows)} Zeilen -> {out}")
