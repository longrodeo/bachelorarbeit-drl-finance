import numpy as np
import pytest
from src.portfolio.vol_controls import VolEWMA, vol_target_step

def test_vol_ewma_updates():
    ew = VolEWMA(span=5)
    sigmas = [ew.update(r) for r in [0.01, -0.02, 0.0, 0.03]]
    assert all(s >= 0 for s in sigmas)
    assert ew.get_sigma() == pytest.approx(sigmas[-1], 1e-12)

def test_vol_target_step_downscale_when_high_vol():
    w_prev = np.array([0.6, 0.4])
    w_tgt  = np.array([0.8, 0.2])  # diff L1 = 0.4
    w_new, scale = vol_target_step(w_prev, w_tgt, vol_estimate=0.20, target_vol=0.10, scaling_limits=(0.5, 2.0))
    # high sigma_hat -> scale ~ 0.5 (clipped), Schritt sollte ~0.2 L1 sein
    l1_step = np.abs(w_new - w_prev).sum()
    assert scale == pytest.approx(0.5, 1e-6)
    assert l1_step == pytest.approx(0.2, 1e-6)
    assert np.all(w_new >= -1e-12) and np.allclose(w_new.sum(), 1.0, atol=1e-12)
    assert np.abs(w_new - w_prev).sum() == pytest.approx(0.5 * 0.4, 1e-6)

def test_vol_target_step_upscale_when_low_vol():
    w_prev = np.array([0.5, 0.5, 0.0])
    w_tgt  = np.array([0.7, 0.3, 0.0])  # diff L1 = 0.4
    w_new, scale = vol_target_step(w_prev, w_tgt, vol_estimate=0.02, target_vol=0.10, scaling_limits=(0.5, 2.0))
    l1_diff = np.abs(w_tgt - w_prev).sum()
    l1_step = np.abs(w_new - w_prev).sum()
    # low sigma_hat -> scale would be 5.0, but clipped to 2.0 ⇒ Schritt ~0.8 L1, aber max 0.4 erreichbar -> voll zum Ziel
    assert scale == pytest.approx(1.0, 1e-6)
    assert np.allclose(w_new, w_tgt)   # weil max Schritt nicht größer als diff ist
    assert l1_step == pytest.approx(l1_diff, 1e-6)