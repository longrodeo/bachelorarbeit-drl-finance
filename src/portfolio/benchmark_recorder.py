import argparse
from pathlib import Path
import pandas as pd

from portfolio.broker import PortfolioLite
import execution, fees
from accounting.recorder import AccountingRecorder
from utils.parquet_io import load_parquet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "data" / "clean" / "benchmark.parquet"
DEFAULT_RF    = ROOT / "data" / "clean" / "riskfree.parquet"


def parse_weights(s: str) -> dict[str, float]:
    # Format: "ACWI=0.85,IAU=0.10,BTC-USD=0.05"
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out


def run(panel_path: Path, rf_path: Path, out_dir: Path, weights_map: dict[str, float],
        per_year: bool, commission_bps: float = 25.0):

    panel = load_parquet(panel_path)  # MultiIndex (date, asset)
    assets_order = panel.index.get_level_values("asset").unique().tolist()

    rf = load_parquet(rf_path)
    if "date" in rf.columns:
        rf = rf.set_index("date")
    rf.index = pd.to_datetime(rf.index).normalize()
    rf = rf.sort_index()
    rf_daily = rf["rf_daily_factor"].astype(float).ffill().bfill()

    dates = pd.to_datetime(panel.index.get_level_values("date").unique()).sort_values()

    # --- Zielgewichte bauen ---
    w_bench = pd.Series(0.0, index=assets_order)
    hit = 0
    for k, v in weights_map.items():
        if k in w_bench.index:
            w_bench.loc[k] = v
            hit += 1

    # Fallback: wenn kein Mapping matched, nimm die ersten 3 Assets
    if hit == 0 and len(assets_order) >= 3:
        w_bench.iloc[0] = 0.85
        w_bench.iloc[1] = 0.10
        w_bench.iloc[2] = 0.05

    # --- Date-Splits ---
    if per_year:
        chunks = []
        for y in sorted(dates.year.unique()):
            d = dates[dates.year == y]
            if len(d) >= 2:
                chunks.append((str(y), d))
    else:
        chunks = [("full", dates)]

    for tag, dates_chunk in chunks:
        pf = PortfolioLite(
            assets=assets_order,
            col_mark="adj_close",
            col_ref="adj_open",
            col_spread="bid_ask_spread_corwin_schultz",
            fee_kwargs={"commission_bps": float(commission_bps), "use_vol_slippage": False},
            execution_mod=execution, fees_mod=fees,
        )

        rec = AccountingRecorder(out_dir / tag)
        last_w = None  # Buy&Hold nach initialem Kauf

        for r in range(len(dates_chunk) - 1):
            t  = dates_chunk[r]
            t1 = dates_chunk[r + 1]

            px_t  = panel.xs(t,  level="date")
            px_t1 = panel.xs(t1, level="date")

            cf = float(rf_daily.reindex([t]).ffill().bfill().iloc[0])

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

        print(f"Benchmark fertig: {panel_path.name} [{tag}] -> {out_dir / tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default=str(DEFAULT_PANEL))
    ap.add_argument("--rf", type=str, default=str(DEFAULT_RF))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--weights", type=str, default="ACWI=0.85,IAU=0.10,BTC-USD=0.05")
    ap.add_argument("--per_year", action="store_true")
    ap.add_argument("--commission_bps", type=float, default=25.0)
    args = ap.parse_args()

    run(
        panel_path=Path(args.panel),
        rf_path=Path(args.rf),
        out_dir=Path(args.out),
        weights_map=parse_weights(args.weights),
        per_year=bool(args.per_year),
        commission_bps=float(args.commission_bps),
    )


if __name__ == "__main__":
    main()
