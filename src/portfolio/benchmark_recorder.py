import pandas as pd
from pathlib import Path

from portfolio.broker import PortfolioLite
import execution, fees
from accounting.recorder import AccountingRecorder
from utils.parquet_io import load_parquet

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "clean" / "benchmark.parquet"
RF_PATH    = ROOT / "data" / "clean" / "riskfree.parquet"

panel = load_parquet(CLEAN_PATH)  # MultiIndex (date, asset)
assets_order = panel.index.get_level_values("asset").unique().tolist()

rf = load_parquet(RF_PATH)
if "date" in rf.columns:
    rf = rf.set_index("date")
rf.index = pd.to_datetime(rf.index).normalize()
rf = rf.sort_index()
rf_daily = rf["rf_daily_factor"].astype(float).ffill().bfill()

dates = pd.to_datetime(panel.index.get_level_values("date").unique()).sort_values()

# Benchmark-Gewichte (falls Ticker abweichen, passt du das Mapping an)
w_bench = pd.Series(0.0, index=assets_order)
target_map = {"ACWI": 0.85, "IAU": 0.10, "BTC-USD": 0.05}  # <- ggf. anpassen

hit = 0
for k, v in target_map.items():
    if k in w_bench.index:
        w_bench.loc[k] = v
        hit += 1

# Fallback: wenn Ticker nicht passen, nimm einfach die ersten 3 Assets
if hit == 0 and len(assets_order) >= 3:
    w_bench.iloc[0] = 0.85
    w_bench.iloc[1] = 0.10
    w_bench.iloc[2] = 0.05

years = sorted(dates.year.unique())

for year in years:
    dates_y = dates[dates.year == year]
    if len(dates_y) < 2:
        continue

    pf = PortfolioLite(
        assets=assets_order,
        col_mark="adj_close",
        col_ref="adj_open",
        col_spread="bid_ask_spread_corwin_schultz",
        fee_kwargs={"commission_bps": 25.0, "use_vol_slippage": False},
        execution_mod=execution, fees_mod=fees,
    )

    rec = AccountingRecorder(ROOT / "data" / "accounting_demo" / f"benchmark_{year}")

    last_w = None  # für Buy&Hold: nach dem initialen Kauf halten wir einfach die bestehenden Gewichte

    for r in range(len(dates_y) - 1):
        t  = dates_y[r]
        t1 = dates_y[r + 1]

        px_t  = panel.xs(t,  level="date")
        px_t1 = panel.xs(t1, level="date")

        cf = float(rf_daily.reindex([t]).ffill().bfill().iloc[0])

        # Jahr-Start: einmal investieren, danach halten
        w_target = w_bench if last_w is None else last_w

        weights_post, info = pf.step(px_t=px_t, px_t1=px_t1, w_target=w_target, cash_factor=cf)
        last_w = weights_post.reindex(assets_order).fillna(0.0)

        rec.log_round(
            t1, assets_order,
            p1=px_t1["adj_close"],
            cash=pf.cash, shares=pf.shares,
            w_post=weights_post,
            exec_df=info["trades"], fees_df=info["fees_detail"],
            round_id=r,
        )

    print(f"Benchmark {year} fertig.")
