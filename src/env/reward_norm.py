# src/env/reward_norm.py
import math

class RewardNormalizer:
    def __init__(self, beta: float = 0.99, clip: float | None = 5.0, warmup: int = 50, eps: float = 1e-8):
        self.beta = beta
        self.clip = clip
        self.warmup = warmup
        self.eps = eps
        self.count = 0
        self.m = 0.0   # EMA-Mean
        self.s = 0.0   # EMA-Second moment

    def reset(self):
        self.count = 0
        self.m = 0.0
        self.s = 0.0

    def __call__(self, r: float) -> float:
        self.count += 1
        b = self.beta
        self.m = b * self.m + (1 - b) * r
        self.s = b * self.s + (1 - b) * (r * r)
        var = max(self.s - self.m * self.m, 0.0)

        if self.count <= self.warmup or var < self.eps:
            z = 0.0  # bis stabil
        else:
            z = (r - self.m) / math.sqrt(var + self.eps)
        if self.clip is not None:
            if z > self.clip:  z = self.clip
            elif z < -self.clip: z = -self.clip
        return z
