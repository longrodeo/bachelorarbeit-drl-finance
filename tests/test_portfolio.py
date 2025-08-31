import numpy as np
import pandas as pd
import math

import portfolio.broker as pf
import portfolio.execution as ex
import portfolio.fees as fs

EPS = 1e-9

def _mk_px(assets, p_t, p_t1, spread_bps=0.0):
    idx = pd.Index(assets, name="asset")
    px_t = pd.Series(p_t, index=idx, dtype=float)  # heutige Markierung (t)
    px_t1 = pd.DataFrame({
        "adj_close": p_t1,                 # Markierung t+1
        "execution_price_t_plus_1_open": p_t1,       # Referenzpreis für Execution (hier = close)
        "bid_ask_spread_corwin_schultz": spread_bps/1e4,   # Corwin-Schultz-Spread in Dezimal
    }, index=idx).astype(float)
    return px_t, px_t1

def test_step_no_costs_matches_targets():
    assets = ["AAA", "BBB"]
    px_t, px_t1 = _mk_px(assets, [100, 200], [110, 220], spread_bps=0.0)
    port = pf.PortfolioLite(
        assets, initial_cash=1_000_000.0,
        execution_mod=ex, fees_mod=fs,
        fee_kwargs=dict(commission_bps=0.0, use_vol_slippage=False),
    )
    w_target = pd.Series([0.6, 0.4], index=px_t.index, dtype=float)

    # Vorwerte
    p1 = px_t1["close"].astype(float)
    Ppre = port.cash + float((port.shares * p1).sum())

    w_post, info = port.step(px_t, px_t1, w_target)

    # 1) Wert-Invariante ohne Kosten: value == Ppre
    assert math.isclose(info["value"], Ppre, rel_tol=0, abs_tol=1e-6)

    # 2) Gewichte: exakt die Zielgewichte (ohne Kosten)
    assert np.allclose(w_post.values, (w_target / w_target.sum()).values, atol=1e-12)
    # 3) Summe 1, nicht-negativ
    assert math.isclose(w_post.sum(), 1.0, abs_tol=1e-12)
    assert (w_post >= -EPS).all()

def test_step_spread_costs_reduce_value_by_half_spread_notional():
    assets = ["AAA", "BBB"]
    # 100bp Spread = 1% ⇒ Half-Spread = 0.5%
    px_t, px_t1 = _mk_px(assets, [100, 200], [100, 200], spread_bps=100.0)
    port = pf.PortfolioLite(
        assets, initial_cash=1_000_000.0,
        execution_mod=ex, fees_mod=fs,
        fee_kwargs=dict(commission_bps=0.0, use_vol_slippage=False),
    )
    w_target = pd.Series([0.5, 0.5], index=px_t.index, dtype=float)

    p1 = px_t1["close"].astype(float)
    Ppre = port.cash + float((port.shares * p1).sum())

    w_post, info = port.step(px_t, px_t1, w_target)

    # Erwartete Spread-Kosten: |q| * p_ref * 0.5 * spread (pro Asset, aufsummiert)
    trades = info["q"].abs() * px_t1["exec_ref_tplus1"] * 0.5 * px_t1["spread_cs"]
    expected_spread = float(trades.sum())

    assert math.isclose(info["value"], Ppre - expected_spread, abs_tol=1e-6)
    assert math.isclose(info["fees"], 0.0, abs_tol=1e-12)

def test_step_commission_bps_applied_and_value_matches_invariant():
    assets = ["AAA"]
    px_t, px_t1 = _mk_px(assets, [100.0], [100.0], spread_bps=0.0)
    port = pf.PortfolioLite(
        assets, initial_cash=1_000_000.0,
        execution_mod=ex, fees_mod=fs,
        fee_kwargs=dict(commission_bps=10.0, use_vol_slippage=False),  # 10 bps = 0.10%
    )
    w_target = pd.Series([1.0], index=px_t.index, dtype=float)

    p1 = px_t1["close"].astype(float)
    Ppre = port.cash + float((port.shares * p1).sum())

    w_post, info = port.step(px_t, px_t1, w_target)

    # Gebühren = 10bps * |q| * p_exec
    q = info["q"].abs().iloc[0]
    p_exec = info["pexec"].iloc[0]
    expected_fees = q * p_exec * (10.0 / 1e4)
    assert math.isclose(info["fees"], expected_fees, rel_tol=1e-12, abs_tol=1e-9)
    assert math.isclose(info["value"], Ppre - expected_fees, abs_tol=1e-6)

def test_execution_half_spread_price_vectorized():
    p_ref = pd.Series([100, 200, 300])
    side  = pd.Series([+1, -1, 0])   # buy, sell, flat
    spread = pd.Series([0.02, 0.01, 0.05])  # 200bps, 100bps, 500bps
    p_exec = ex.half_spread_price(p_ref, side, spread)
    assert np.isclose(p_exec.iloc[0], 100 * (1 + 0.5*0.02))  # buy → + half spread
    assert np.isclose(p_exec.iloc[1], 200 * (1 - 0.5*0.01))  # sell → − half spread
    assert np.isclose(p_exec.iloc[2], 300)  # side==0 → treated ≥0

def test_fees_total_cost_column_is_sum():
    trades = pd.DataFrame({
        "q": [10, -5],
        "p_ref": [100.0, 200.0],
        "p_exec": [101.0, 199.0],
        "notional_abs": [10*101.0, 5*199.0],
        "spread_cost": [10*100.0*0.5*0.01, 5*200.0*0.5*0.02],  # 100bps & 200bps Half-Spread
    }, index=["AAA", "BBB"])
    out = fs.apply_fees(trades, commission_bps=5.0, use_vol_slippage=False)
    assert "fees" in out and "vol_slip" in out and "total_cost" in out
    expected = out["spread_cost"] + out["fees"] + out["vol_slip"]
    assert np.allclose(out["total_cost"].values, expected.values)

def test_weights_sum_to_one_and_clip_when_no_short():
    assets = ["AAA", "BBB", "CCC"]
    px_t, px_t1 = _mk_px(assets, [100, 100, 100], [100, 100, 100], spread_bps=0.0)
    port = pf.PortfolioLite(
        assets, initial_cash=100_000.0,
        execution_mod=ex, fees_mod=fs, fee_kwargs=dict(),
        # allow_short=False ist Default
    )
    w_target = pd.Series([0.5, -0.2, 0.7], index=px_t.index, dtype=float)  # negative Gewicht wird geclippt
    w_post, _ = port.step(px_t, px_t1, w_target)
    assert (w_post >= -EPS).all()
    assert math.isclose(float(w_post.sum()), 1.0, abs_tol=1e-12)
