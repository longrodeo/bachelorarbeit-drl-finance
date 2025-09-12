# src/envs/data_builder.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List, Dict
import pandas as pd

from src.data.load_panel_years import load_panel_years
from src.env.trading_env import TradingEnv
from src.accounting.recorder import AccountingRecorder

Segment = Tuple[str, str]

def load_data_for_windows(windows: Iterable[Dict[str, List[Segment]]]):
    years: set[int] = set()
    for w in windows:
        for seg in w.get("train", []) + w.get("test", []):
            years.add(int(seg[0][:4])); years.add(int(seg[1][:4]))
    return load_panel_years(sorted(years))

def _find_indices(dates: pd.DatetimeIndex, start_dt: str, end_dt: str) -> tuple[int, int]:
    s = dates.slice_indexer(pd.Timestamp(start_dt), pd.Timestamp(end_dt), step=None)
    # slice_indexer liefert [start:stop) – stop ist EXKLUSIV, wenn end_dt < next(date)
    return int(s.start), int(s.stop)
def build_env_segment(panel, seg: Segment, *, state_spec: str, reward_kind: str,
                      with_recorder: bool, out_dir: Path | None):
    dates = panel["dates"]  # Erwartet: aufsteigend sortiertes Array/Index der Datumsstrings
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)
    start_idx, end_idx_excl = _find_indices(dates, seg[0], seg[1])
    rec = AccountingRecorder(out_dir=Path(out_dir)) if (with_recorder and out_dir is not None) else None
    env = TradingEnv(
        panel=panel,
        state_spec_path=state_spec,
        reward_kind=reward_kind,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_excl,
        recorder=rec,
    )
    return env
