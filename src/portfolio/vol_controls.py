from __future__ import annotations
import numpy as np

class VolEWMA:
    """
    Ex-ante Volatilitätsschätzer (EWMA) über Portfolio-Returns.
    sigma_t = sqrt(lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2)
    - Leak-frei: r_t ist realisiert (bis t) und wird für t+1 genutzt.
    """
    def __init__(self, span: int = 60, eps: float = 1e-12):
        if span <= 1:
            raise ValueError("span must be > 1")
        # alpha = 2/(span+1); lambda = 1 - alpha
        self.alpha = 2.0 / (span + 1.0)           # Gewicht für neue Beobachtung
        self.lambda_ = 1.0 - self.alpha           # Gewicht für Historie
        self.var = 0.0
        self.sigma = 0.0
        self.eps = eps
        self._warm = False

    def update(self, portfolio_return_t: float) -> float:
        """Update mit realisiertem (daily) Portfolio-Return r_t (z. B. von t-1→t)."""
        r2 = float(portfolio_return_t) * float(portfolio_return_t)
        if not self._warm:
            self.var = r2
            self._warm = True
        else:
            self.var = self.lambda_ * self.var + self.alpha * r2
        self.sigma = float(np.sqrt(max(self.var, 0.0)))
        return self.sigma

    def get_sigma(self) -> float:
        return float(self.sigma)


def vol_target_step(
    weights_prev: np.ndarray,
    weights_target: np.ndarray,
    vol_estimate: float,
    target_vol: float = 0.10,                # Zielvola (gleiche Einheit wie vol_estimate!)
    scaling_limits: tuple[float, float] = (0.5, 2.0),
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Skaliert NUR die Schrittweite (weights_target - weights_prev) mit einem Faktor:
        scale_unclipped = target_vol / (vol_estimate + eps)
        scale_clipped   = clip(scale_unclipped, scaling_limits)

    WICHTIG:
    - Für die Gewichte wird niemals "über das Ziel hinaus" gegangen:
      applied_scale = min(scale_clipped, 1.0)
    - Die Funktion GIBT den Wert 'scale_clipped' zurück (historisch kompatibel),
      nicht den angewandten 'applied_scale'.

    Returns:
        weights_scaled : Gewichte nach Skalierung & Renormalisierung (Simplex)
        scale_clipped  : Geklipptes Verhältnis target_vol / vol_estimate (zur Info/Logs)

    Hinweis zu Einheiten:
    - Wenn deine Returns daily sind, ist vol_estimate daily. Dann target_vol bitte auch
      daily liefern (z. B. 0.10/√252). Wichtig ist nur: BEIDE gleich skaliert.
    """
    weights_prev = np.asarray(weights_prev, dtype=float)
    weights_target = np.asarray(weights_target, dtype=float)
    if weights_prev.shape != weights_target.shape:
        raise ValueError("weights_prev and weights_target must have same shape")
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")

    scale_unclipped = target_vol / (float(vol_estimate) + eps)
    scale_clipped = float(np.clip(scale_unclipped, scaling_limits[0], scaling_limits[1]))

    # nie über das Ziel hinaus
    applied_scale = min(scale_clipped, 1.0)

    step = weights_target - weights_prev
    weights_scaled = weights_prev + applied_scale * step

    # Simplex-Sicherung
    weights_scaled = np.maximum(weights_scaled, 0.0)
    s = float(weights_scaled.sum())
    if s > eps:
        weights_scaled /= s
    else:
        weights_scaled = np.zeros_like(weights_scaled)
        weights_scaled[0] = 1.0

    return weights_scaled, applied_scale
