# src/envs/data_builder.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Dict
import pandas as pd

import src.state.state_builder as sb
from src.data.load_panel_years import load_panel_years, load_panel_wf
from src.env.trading_env import TradingEnv
from src.accounting.recorder import AccountingRecorder
from src.utils.paths import get_assets_flat, get_asset_groups
from src.portfolio.broker import PortfolioLite

Segment = Tuple[str, str]

def load_data_for_windows(windows: Iterable[Dict[str, List[Segment]]],
                          strategy: str,
                          features_source: str | None = None):
    """
    CPCV: lädt nur die benötigten Jahre.
    Walk-Forward (expanding): lädt ein komplettes Feature-Panel (features_source).
    """
    if strategy == "walkforward":
        if not features_source:
            features_source = "features_v1_raw_z"
        return load_panel_wf(features_source)  # EIN File für WF
    # sonst: CPCV
    years: set[int] = set()
    for w in windows:
        for seg in w.get("train", []) + w.get("test", []):
            years.add(int(seg[0][:4])); years.add(int(seg[1][:4]))
    return load_panel_years(sorted(years))

def _find_indices(dates: pd.DatetimeIndex, start_dt: str, end_dt: str) -> tuple[int, int]:
    s = dates.slice_indexer(pd.Timestamp(start_dt), pd.Timestamp(end_dt), step=None)
    # slice_indexer liefert [start:stop) – stop ist EXKLUSIV, wenn end_dt < next(date)
    return int(s.start), int(s.stop)

def build_env_segment(panel, seg: Segment, *, state_spec: str, reward_kind: str, initial_cash: float = 1_000_000.0,
                      with_recorder: bool, out_dir: Path | None):
    dates = panel["dates"]  # Erwartet: aufsteigend sortiertes Array/Index der Datumsstrings
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    rf_factor = panel["rf_daily_factor_raw"].reindex(dates).to_numpy()
    rf_rate = panel["risk_free_rate_raw"].reindex(dates).to_numpy()
    assets = get_assets_flat(get_asset_groups())
    start_idx, end_idx_excl = _find_indices(dates, seg[0], seg[1])
    rec = AccountingRecorder(out_dir=Path(out_dir)) if (with_recorder and out_dir is not None) else None
    env = TradingEnv(
        panel_clean=panel,
        panel_features=panel,
        assets=assets,
        dates=dates,
        state_builder=sb,
        spec=state_spec,
        portfolio=PortfolioLite(assets=assets, initial_cash=initial_cash),
        initial_cash=initial_cash,
        reward_kind=reward_kind,
        rf_rate=rf_rate,
        rf_factor=rf_factor,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_excl,
        recorder=rec,
    )
    return env
