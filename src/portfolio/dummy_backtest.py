import pandas as pd
from pathlib import Path

from portfolio.broker import PortfolioLite
import execution, fees
from accounting.recoder import AccountingRecorder
from utils.parquet_io import load_parquet
from src.features.riskfree_interest import daily_factor

# 1) Data & Setup
ROOT = Path(__file__).resolve().parents[2]  # eine Ebene hoch von tests/
CLEAN_PATH = ROOT / "data" / "clean" / "features_v1.parquet"
panel = pd.read_parquet(CLEAN_PATH)  # MultiIndex (date, asset)
assets_order = panel.index.get_level_values("asset").unique().tolist()
assets_order = [a for a in assets_order if a != "CASH"]

RF_PATH = ROOT / "data" / "clean" / "riskfree.parquet"  # <- Pfad ggf. anpassen
rf = load_parquet(RF_PATH)

# --- Zinsdaten vorbereiten (Kalendertags-Auflösung + Tagesfaktoren) ---
if "date" in rf.columns:
    rf = rf.set_index("date")

rf.index = pd.to_datetime(rf.index).normalize()
rf = rf.sort_index()


rf_ann_pct = rf["risk_free_annual"]

rf_ann_pct = rf_ann_pct.astype(float).ffill()

# 2) Tagesfaktoren aus Jahreszins (Basis 360), dann auf Kalendertage auffüllen
f_daily = daily_factor(rf_ann_pct, basis=360)   # -> 1 + r_daily
f_daily = f_daily.asfreq("D", method="ffill")

# 3) Faktor für Intervall [t, t1): Produkt der Tagesfaktoren
from datetime import timedelta
def cash_factor(t):
    t = pd.Timestamp(t).normalize()
    return f_daily[t]

# (Optional) Sanity-Check: alle Handelstage im f_daily-Index
# assert pd.to_datetime(dates).normalize().isin(f_daily.index).all()


dates = panel.index.get_level_values("date").unique().sort_values()
dates = dates[1:501]   # nimm die ersten 10 Handelstage

if "date" in rf.columns:
    rf = rf.set_index("date")
rf.index = pd.to_datetime(rf.index)
rf = rf.sort_index()

pf = PortfolioLite(
    assets=assets_order,
    col_mark="adj_close",
    col_ref="adj_open",
    col_spread="bid_ask_spread_corwin_schultz",
    fee_kwargs={"commission_bps": 3.0, "use_vol_slippage": False},
    execution_mod=execution, fees_mod=fees,
)

rec = AccountingRecorder(ROOT / "data" / "accounting_demo")

# 2) Loop (5 Runden)
for r in range(10):
    t = dates[r]
    px_t  = panel.xs(t, level="date")
    # px_t1 = panel.xs(t1, level="date")

    # Dummy-Agent: rotiert zufällig durch Assets
    w_target = pd.Series(0.0, index=assets_order)
    if r % 2 == 0:
        w_target.iloc[2] = 0.50   # alle 2 Runden ins erste Asset
        w_target.iloc[3] = 0.50
    else:
        w_target.iloc[2] = 0.50   # sonst ins zweite Asset
        w_target.iloc[3] = 0.50

    cf = cash_factor(t)
    print(cf)
    if r >= 1:
        weights_post, info = pf.step(px_t1=px_t, w_target=w_target, cash_factor=cf)

    # Accounting loggen
        rec.log_round(
            t, assets_order,
            p1=px_t["adj_close"],
            cash=pf.cash, shares=pf.shares,
            w_post=weights_post,
            exec_df=info["trades"], fees_df=info["fees_detail"],
            round_id=r,
        )
    if r < 1:
        w_target = pd.Series(0.0, index=assets_order)
        weights_post, info = pf.step(px_t1=px_t, w_target=w_target, cash_factor=cf)

        rec.log_round(
            t, assets_order,
            p1=px_t["adj_close"],
            cash=pf.cash, shares=pf.shares,
            w_post=weights_post,
            exec_df=info["trades"], fees_df=info["fees_detail"],
            round_id=r,
        )

print("Fünf Runden fertig – schau in data/accounting_demo/")
