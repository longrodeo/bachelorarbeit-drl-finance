import pandas as pd
from pathlib import Path

from portfolio.broker import PortfolioLite
import execution, fees
from accounting.recorder import AccountingRecorder
from utils.parquet_io import load_parquet

# 1) Data & Setup
ROOT = Path(__file__).resolve().parents[2]  # eine Ebene hoch von tests/
CLEAN_PATH = ROOT / "data" / "clean" / "benchmark.parquet"
panel = load_parquet(CLEAN_PATH)  # MultiIndex (date, asset)
assets_order = panel.index.get_level_values("asset").unique().tolist()

RF_PATH = ROOT / "data" / "clean" / "riskfree.parquet"  # <- Pfad ggf. anpassen
rf = load_parquet(RF_PATH)

# --- Zinsdaten vorbereiten (Kalendertags-Auflösung + Tagesfaktoren) ---
if "date" in rf.columns:
    rf = rf.set_index("date")

rf.index = pd.to_datetime(rf.index).normalize()
rf = rf.sort_index()


rf_daily = rf["rf_daily_factor"]
rf_daily = rf_daily.ffill().bfill()

# (Optional) Sanity-Check: alle Handelstage im f_daily-Index
# assert pd.to_datetime(dates).normalize().isin(f_daily.index).all()


dates = panel.index.get_level_values("date").unique().sort_values()
print(len(dates))
dates = dates[0:2515]   # nimm die ersten 10 Handelstage


if "date" in rf.columns:
    rf = rf.set_index("date")
rf.index = pd.to_datetime(rf.index)
rf = rf.sort_index()

pf = PortfolioLite(
    assets=assets_order,
    col_mark="adj_close",
    col_ref="adj_open",
    col_spread="bid_ask_spread_corwin_schultz",
    fee_kwargs={"commission_bps": 25.0, "use_vol_slippage": False},
    execution_mod=execution, fees_mod=fees,
)

rec = AccountingRecorder(ROOT / "data" / "accounting_demo")

# 2) Loop (5 Runden)
for r in range(2515):
    t1 = dates[r+1]
    t = dates[r]
    px_t1  = panel.xs(t1, level="date")
    px_t = panel.xs(t, level="date")

    # Dummy-Agent: rotiert zufällig durch Assets
    w_target = pd.Series(0.0, index=assets_order)
    if r % 2 == 0:
        w_target.iloc[0] = 0.85   # alle 2 Runden ins erste Asset
        w_target.iloc[1] = 0.05
        w_target.iloc[2] = 0.10

    else:
        w_target.iloc[0] = 0.85
        w_target.iloc[1] = 0.05
        w_target.iloc[2] = 0.10

    cf = rf_daily.reindex([t]).ffill().bfill().iloc[0]
    cf = float(cf)

    if r >= 0:
        weights_post, info = pf.step(px_t = px_t, px_t1=px_t1, w_target=w_target, cash_factor=cf)
        #print(px_t)
        # assert not px_t.isna().any().any(), f"NaN im Preis bei {t}"
        assert not pd.isna(cf), f"NaN im riskfree bei {t}"

    # Accounting loggen
        rec.log_round(
            t1, assets_order,
            p1=px_t1["adj_close"],
            cash=pf.cash, shares=pf.shares,
            w_post=weights_post,
            exec_df=info["trades"], fees_df=info["fees_detail"],
            round_id=r,
        )
print("Fünf Runden fertig – schau in data/accounting_demo/")
