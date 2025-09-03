# src/portfolio/vol_controls.py
from __future__ import annotations
import numpy as np

class VolEWMA:
    """
    Ex-ante Volatilitätsschätzer (EWMA) über Portfolio-Returns.
    sigma_t = sqrt(lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2)
    - Kein Leak: r_t ist realisiert (bis t), wird für Trade-Entscheidung an t+1 genutzt.
    """
    def __init__(self, span: int = 60, eps: float = 1e-12):
        if span <= 1: raise ValueError("span must be > 1")
        self.alpha = 2.0 / (span + 1.0)           # 1 - lambda
        self.one_minus_alpha = 1.0 - self.alpha   # lambda
        self.var = 0.0
        self.sigma = 0.0
        self.eps = eps
        self._warm = False

    def update(self, r_t: float) -> float:
        """Update mit realisiertem Portfolio-Return r_t (z. B. von t-1→t)."""
        r2 = float(r_t) * float(r_t)
        if not self._warm:
            self.var = r2
            self._warm = True
        else:
            self.var = self.one_minus_alpha * self.var + self.alpha * r2
        self.sigma = float(np.sqrt(max(self.var, 0.0)))
        return self.sigma

    def get_sigma(self) -> float:
        return float(self.sigma)

def vol_target_step(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    sigma_hat: float,
    sigma_tgt: float = 0.10,          # jährliche Zielvola, ggf. parametrisierbar
    clip: tuple[float, float] = (0.5, 2.0),
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Skaliert NUR die Schrittweite (w_target - w_prev) mit scale = clip(sigma_tgt / (sigma_hat + eps)).
    - Ändert NICHT die Reward-Definition.
    - Bewahrt Richtung der Aktion, dämpft/verstärkt nur die Größe.
    - Renormalisiert auf Simplex (long-only).
    Returns: (w_scaled_target, scale)
    """
    w_prev = np.asarray(w_prev, dtype=float)
    w_target = np.asarray(w_target, dtype=float)
    if w_prev.shape != w_target.shape:
        raise ValueError("w_prev and w_target must have same shape")
    if sigma_tgt <= 0: raise ValueError("sigma_tgt must be > 0")

    # annual → per-step? Wenn deine Returns daily sind: sigma_tgt_daily ≈ sigma_tgt / sqrt(252)
    # Für Einfachheit: hier sigma_tgt als "per-step" anliefern. (Sonst vorher umrechnen.)
    scale_raw = sigma_tgt / (float(sigma_hat) + eps)
    scale = float(np.clip(scale_raw, clip[0], clip[1]))

    # >>> Sättigung: nie über das Ziel hinaus
    applied_scale = min(scale, 1.0)

    d = w_target - w_prev
    w_scaled = w_prev + applied_scale * d

    # Simplex-Sicherung
    w_scaled = np.maximum(w_scaled, 0.0)
    s = float(w_scaled.sum())
    if s > eps:
        w_scaled /= s
    else:
        w_scaled = np.zeros_like(w_scaled); w_scaled[0] = 1.0

    return w_scaled, scale
