import numpy as np
import pytest
from src.portfolio.execution_controls import (no_trade_band, apply_turnover_cap, apply_execution_controls)

def test_no_trade_band_no_action():
    w_prev, w_target = np.array([0.5,0.5,0.0]), np.array([0.51,0.49,0.0])  # L1=0.02
    w_exec, l1 = no_trade_band(w_prev, w_target, delta_l1=0.03)
    assert np.allclose(w_exec, w_prev)
    assert l1 == pytest.approx(0.02, rel=1e-6, abs=1e-12)

def test_no_trade_band_action():
    w_prev, w_target = np.array([0.5,0.5,0.0]), np.array([0.65,0.35,0.0])  # L1=0.30
    w_exec, l1 = no_trade_band(w_prev, w_target, delta_l1=0.03)
    assert np.allclose(w_exec, w_target)
    assert l1 == pytest.approx(0.30, rel=1e-6, abs=1e-12)

def test_turnover_cap_no_limit_needed():
    w_prev, w_target = np.array([0.6,0.4]), np.array([0.7,0.3])  # L1=0.2
    w_exec, l1_req, scale = apply_turnover_cap(w_prev, w_target, tau_l1=0.3)
    assert np.allclose(w_exec, w_target)
    assert l1_req == pytest.approx(0.2, 1e-6)
    assert scale == pytest.approx(1.0, 1e-9)
    assert np.allclose(w_exec.sum(), 1.0)

def test_turnover_cap_limits_step():
    w_prev, w_target = np.array([0.5,0.5,0.0]), np.array([0.9,0.1,0.0])  # L1=0.8
    w_exec, l1_req, scale = apply_turnover_cap(w_prev, w_target, tau_l1=0.2)
    # Der tatsächliche Schritt sollte ~0.2 (L1) sein
    step = np.abs(w_exec - w_prev).sum()
    assert l1_req == pytest.approx(0.8, 1e-6)
    assert scale == pytest.approx(0.25, 1e-6)  # 0.2 / 0.8
    assert step == pytest.approx(0.2, 1e-6)
    # Simplex & Nonnegativity
    assert np.all(w_exec >= -1e-12)
    assert np.allclose(w_exec.sum(), 1.0, atol=1e-12)

def test_turnover_cap_zero_tau_returns_prev():
    w_prev, w_target = np.array([0.2,0.8]), np.array([0.9,0.1])
    w_exec, l1_req, scale = apply_turnover_cap(w_prev, w_target, tau_l1=0.0)
    assert np.allclose(w_exec, w_prev)
    assert scale == pytest.approx(0.0, 1e-12)

def test_apply_execution_controls_no_trade():
    w_prev = np.array([0.5, 0.5, 0.0])
    w_tgt = np.array([0.51, 0.49, 0.0])  # L1=0.02 < 0.03
    w_new, info = apply_execution_controls(w_prev, w_tgt, delta_l1=0.03, tau_l1=0.2)
    assert np.allclose(w_new, w_prev)
    assert info["acted"] == 0.0
    assert info["l1_step"] == pytest.approx(0.0)

def test_apply_execution_controls_with_cap():
    w_prev = np.array([0.5, 0.5, 0.0])
    w_tgt = np.array([0.9, 0.1, 0.0])  # L1_req=0.8 > tau=0.2
    w_new, info = apply_execution_controls(w_prev, w_tgt, delta_l1=0.01, tau_l1=0.2)
    assert info["acted"] == 1.0
    assert info["l1_req"] == pytest.approx(0.8, 1e-6)
    assert info["scale"] == pytest.approx(0.25, 1e-6)
    assert info["l1_step"] == pytest.approx(0.2, 1e-6)
    assert np.all(w_new >= -1e-12)
    assert np.allclose(w_new.sum(), 1.0, atol=1e-12)

def test_apply_execution_controls_no_cap_needed():
    w_prev = np.array([0.6, 0.4])
    w_tgt = np.array([0.7, 0.3])  # L1_req=0.2 <= tau=0.3
    w_new, info = apply_execution_controls(w_prev, w_tgt, delta_l1=0.01, tau_l1=0.3)
    assert np.allclose(w_new, w_tgt)
    assert info["scale"] == pytest.approx(1.0, 1e-9)