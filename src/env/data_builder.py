# src/envs/data_builder.py
from __future__ import annotations
from typing import Iterable, Tuple, List, Dict
import pandas as pd
from types import SimpleNamespace
import yaml
import os
from pathlib import Path

from src.rl.wrapper import ActionMappingWrapper
from stable_baselines3.common.monitor import Monitor
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

def _load_state_spec(spec):
    if isinstance(spec, str) and os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # Robust: fehlende Keys als None/[] setzen
        cfg.setdefault("per_asset_features", [])
        cfg.setdefault("global_features", [])
        cfg.setdefault("mask_feature", None)
        return SimpleNamespace(**cfg)
    return spec

def _find_indices(dates, start_dt: str, end_dt: str):
    s = pd.to_datetime(start_dt, utc=True).tz_convert(None)
    e = pd.to_datetime(end_dt,   utc=True).tz_convert(None)
    sl = dates.slice_indexer(s, e, step=None)
    return int(sl.start), int(sl.stop)

def build_env_segment(panel, seg, state_spec, reward_kind, with_recorder, out_dir):


    # 1) Spec sicher als Objekt laden
    if isinstance(state_spec, (str, os.PathLike, Path)):
        state_spec = sb.load_spec(str(state_spec))


    # 2) Dates tz-konsistent & sortiert
    dates = panel.index.get_level_values(0).unique().sort_values()
    dates = pd.to_datetime(dates, utc=True).tz_convert(None)  # tz-naiv wie empfohlen

    # 3) Fenster-Indices aus seg ("YYYY-MM-DD", "YYYY-MM-DD")
    start_s, end_s = seg
    start_dt = pd.to_datetime(start_s, utc=True).tz_convert(None)
    end_dt   = pd.to_datetime(end_s,   utc=True).tz_convert(None)
    start_idx = int(dates.searchsorted(start_dt, side="left"))
    end_idx_exclusive = int(dates.searchsorted(end_dt + pd.Timedelta(days=1), side="left"))
    assert 0 <= start_idx < end_idx_exclusive <= len(dates)

    # 4) Assets & rf_* wie im Probe-Runner
    assets = get_assets_flat(get_asset_groups())

    rf_factor = panel["rf_daily_factor_raw"].groupby(level=0).first()
    rf_rate   = panel["risk_free_rate_raw"].groupby(level=0).first()
    rf_factor.index = pd.to_datetime(rf_factor.index, utc=True).tz_convert(None)
    rf_rate.index   = pd.to_datetime(rf_rate.index,   utc=True).tz_convert(None)
    rf_factor = rf_factor.reindex(dates).to_numpy()
    rf_rate   = rf_rate  .reindex(dates).to_numpy()

    recorder = None
    if with_recorder:
        out_dir = Path(out_dir) if out_dir else Path("data/accounting/tmp")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "segment_start": str(start_dt.date()),
            "segment_end":   str(end_dt.date()),
            "reward_kind":   reward_kind,
            "assets":        list(assets),
        }
        recorder = AccountingRecorder(out_dir=out_dir, meta=meta)

    # 5) Env exakt wie im Probe-Runner instanziieren
    env = TradingEnv(
        panel_clean=panel,
        panel_features=panel,
        dates=dates,
        assets=assets,
        spec=state_spec,
        state_builder=sb,
        portfolio=PortfolioLite(assets=assets, initial_cash=1_000_000.0),
        initial_cash=1_000_000.0,
        rf_factor=rf_factor,
        rf_rate=rf_rate,
        reward_kind=reward_kind,
        recorder=recorder,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_exclusive,
        validate_actions=True,
    )

    # 6) Wrapper wie im Probe-Runner
    env = ActionMappingWrapper(env)
    env = Monitor(env)
    return env
