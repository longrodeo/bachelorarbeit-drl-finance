from pathlib import Path
import pandas as pd
from src.utils.parquet_io import load_parquet, save_parquet

class AccountingRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snap_path = self.out_dir / "portfolio_snapshots.parquet"
        self.evt_path  = self.out_dir / "trade_events.parquet"

    def _append(self, df: pd.DataFrame, path: Path):
        if path.exists():
            old = load_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        save_parquet(df, path)

    def log_round(self, t, assets, p1, cash, shares,
                  w_post, exec_df, fees_df, round_id: int):
        # Snapshots
        snap = pd.DataFrame({
            "round": round_id, "t": t,
            "asset": assets,
            "p_close_t": p1.reindex(assets).values,
            "shares": shares.reindex(assets).values,
            "weight_post_t": w_post.reindex(assets).values,
            "cash": cash,
            "portfolio_value_t": float((shares*p1).sum() + cash),
            "fees_total_round": float(fees_df["total_cost"].sum()),
        })
        self._append(snap, self.snap_path)

        # Events (nur q != 0)
        evt = exec_df.join(fees_df[["fees","vol_slip","total_cost"]], how="left")
        evt = evt.reset_index(names=["asset"])
        evt["round"] = round_id; evt["t"] = t
        evt = evt.loc[evt["q"] != 0]
        if not evt.empty:
            self._append(evt, self.evt_path)
