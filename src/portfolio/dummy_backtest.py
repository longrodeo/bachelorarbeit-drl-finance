import pandas as pd
from pathlib import Path
from portfolio.broker import PortfolioLite
import execution, fees
from accounting.recoder import AccountingRecorder
from utils.parquet_io import load_parquet


# 1) Data & Setup
ROOT = Path(__file__).resolve().parents[2]  # eine Ebene hoch von tests/
CLEAN_PATH = ROOT / "data" / "clean" / "features_v1.parquet"
panel = pd.read_parquet(CLEAN_PATH)  # MultiIndex (date, asset)
assets_order = panel.index.get_level_values("asset").unique().tolist()

dates = panel.index.get_level_values("date").unique().sort_values()
dates = dates[:11]   # nimm die ersten 10 Handelstage

pf = PortfolioLite(
    assets=assets_order,
    col_mark="adj_close",
    col_ref="execution_price_t_plus_1_open",
    col_spread="bid_ask_spread_corwin_schultz",
    fee_kwargs={"commission_bps": 3.0, "use_vol_slippage": False},
    execution_mod=execution, fees_mod=fees,
)

rec = AccountingRecorder(ROOT / "data" / "accounting_demo")

# 2) Loop (5 Runden)
for r in range(10):
    t, t1 = dates[r], dates[r+1]
    px_t  = panel.xs(t, level="date")["adj_close"]
    px_t1 = panel.xs(t1, level="date")

    # Dummy-Agent: rotiert zufällig durch Assets
    w_target = pd.Series(0.0, index=assets_order)
    if r % 2 == 0:
        w_target.iloc[0] = 1.0   # alle 2 Runden ins erste Asset
    else:
        w_target.iloc[1] = 1.0   # sonst ins zweite Asset

    weights_post, info = pf.step(px_t1=px_t1, w_target=w_target)

    # Accounting loggen
    rec.log_round(
        t, t1, assets_order,
        p1=px_t1["adj_close"],
        cash=pf.cash, shares=pf.shares,
        w_post=weights_post,
        exec_df=info["trades"], fees_df=info["fees_detail"],
        round_id=r+1,
    )

print("Fünf Runden fertig – schau in data/accounting_demo/")
