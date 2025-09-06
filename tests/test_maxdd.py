# tests/test_mdd_delta_with_evaluator.py
import numpy as np
import pandas as pd
from accounting.evaluator import _mdd_series  # nutzt (peak - nav) / peak

def test_delta_mdd_basic_uses_evaluator_logic():
    nav = pd.Series([100, 90, 95, 80, 120, 110], dtype=float)

    # exakt wie in compute_rewards_from_snapshots:
    mdd_t  = _mdd_series(nav.shift(1).ffill())
    mdd_t1 = _mdd_series(nav.ffill())
    delta  = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)

    # Invarianten + erwartete Δ-MDD
    assert np.all((mdd_t1 >= 0) & (mdd_t1 <= 1 + 1e-12))
    expected = np.array([0.0, 0.10, 0.0, 0.15, 0.0, 0.0833])
    assert np.allclose(delta.values.round(4), expected.round(4))

def test_delta_mdd_no_penalty_on_new_highs_or_flats():
    nav = pd.Series([100, 105, 110, 110, 111], dtype=float)
    mdd_t  = _mdd_series(nav.shift(1).ffill())
    mdd_t1 = _mdd_series(nav.ffill())
    delta  = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)
    assert float(delta.sum()) == 0.0  # neue Hochs/Flat erhöhen MDD nicht

def test_delta_mdd_initial_nan_handling_matches_fillna_zero():
    nav = pd.Series([np.nan, 100, 99], dtype=float)
    mdd_t  = _mdd_series(nav.shift(1).ffill())
    mdd_t1 = _mdd_series(nav.ffill())
    delta  = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)
    assert delta.iloc[0] == 0.0

def test_delta_mdd_all_up_is_zero():
    nav = pd.Series([100, 101, 103, 110], dtype=float)
    mdd_t  = _mdd_series(nav.shift(1).ffill())
    mdd_t1 = _mdd_series(nav.ffill())
    delta  = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)
    assert float(delta.sum()) == 0.0  # nur neue Hochs → keine Δ-Strafe

def test_delta_mdd_monotone_down_increments():
    nav = pd.Series([100, 90, 80, 70], dtype=float)
    mdd_t  = _mdd_series(nav.shift(1).ffill())
    mdd_t1 = _mdd_series(nav.ffill())
    delta  = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)
    expected = np.array([0.0, 0.10, 0.10, 0.10])  # jedes neue Tief erhöht Δ
    assert np.allclose(delta.values.round(4), expected.round(4))