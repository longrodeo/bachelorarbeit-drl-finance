# tests/test_reward_norm.py
from src.env.reward_norm import RewardNormalizer
import numpy as np

def test_warmup_zero():
    rn = RewardNormalizer(warmup=5)
    vals = [rn(1.0) for _ in range(5)]
    assert all(v == 0.0 for v in vals)

def test_unit_variance_after_warmup():
    np.random.seed(0)
    rn = RewardNormalizer(beta=0.98, warmup=50, clip=None)
    zs = [rn(np.random.randn()*0.5 + 0.2) for _ in range(500)]
    tail = np.array(zs[200:])
    v = tail.var()
    assert 0.3 < v < 3.0
