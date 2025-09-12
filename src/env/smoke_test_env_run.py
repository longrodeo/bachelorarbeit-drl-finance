# src/run_smoke_env.py
import pandas as pd
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from src.utils.paths import SPEC_S0_YAML
from src.env.smoke_test_env import main
from src.data.load_panel_years import load_panel_years

# — wähle 1–2 Jahre für den Smoke —
panel = load_panel_years([2015, 2016])
panel_clean = panel_features = panel

# Zeitachse & Assets
dates  = panel.index.get_level_values("date").unique().sort_values()
assets = panel.index.get_level_values("asset").unique().tolist()

# Risk-free direkt aus dem Panel
rf_factor = panel.groupby(level="date")["rf_daily_factor_raw"].first().to_numpy()
rf_rate   = panel.groupby(level="date")["risk_free_rate_z"].first().to_numpy()


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
