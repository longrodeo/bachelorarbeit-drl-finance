from typing import Tuple, Dict
import numpy as np

def no_trade_band(w_prev: np.ndarray, w_target: np.ndarray, delta_l1: float = 0.03) -> Tuple[np.ndarray, float]:
    """
    Rebalance nur, wenn sich Zielgewichte spürbar ändern.
    - L1-Abstand (sum(abs(diff))) > delta_l1 → handeln, sonst nix.
    """
    w_prev = np.asarray(w_prev, dtype=float)
    w_target = np.asarray(w_target, dtype=float)
    if w_prev.shape != w_target.shape:
        raise ValueError("w_prev and w_target müssen die gleiche Form haben")
    l1 = float(np.abs(w_target - w_prev).sum())
    if l1 > delta_l1:
        return w_target, l1  # handeln
    else:
        return w_prev, l1  # nichts tun

def apply_turnover_cap(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    tau_l1: float = 0.30,
    eps: float = 1e-12
) -> Tuple[np.ndarray, float, float]:
    """
    Begrenze den Rebalance-Schritt (L1-Norm) auf tau_l1.
    - Wenn L1(w_target - w_prev) <= tau_l1: gehe voll zu w_target (scale=1).
    - Sonst: gehe nur anteilig in Richtung w_target mit scale = tau_l1 / L1(...).
    Hinweis:
    - Liegen w_prev & w_target auf dem Simplex (>=0, Summe=1), bleibt w_exec das auch
      (Konvexkombination). Kleine numerische Abweichungen werden renormalisiert.

    Returns:
        w_exec : ausgeführte Gewichte nach Cap
        l1_req : ursprünglicher L1(w_target - w_prev)
        scale  : tatsächlich verwendeter Schrittfaktor in [0,1]
    """
    w_prev = np.asarray(w_prev, dtype=float)
    w_target = np.asarray(w_target, dtype=float)
    if w_prev.shape != w_target.shape:
        raise ValueError("w_prev and w_target müssen die gleiche Form haben")
    if tau_l1 < 0:
        raise ValueError("tau_l1 muss >= 0")

    diff = w_target - w_prev
    l1_req = float(np.abs(diff).sum())

    if l1_req <= tau_l1 + eps or l1_req <= eps or tau_l1 >= 2.0 - eps:
        w_exec = w_target.copy()
        scale = 1.0
    else:
        scale = float(tau_l1 / l1_req)
        w_exec = w_prev + scale * diff

    # numerische Sanity: auf Simplex zurück (falls Minusrundungen / Summen-Drift)
    w_exec = np.maximum(w_exec, 0.0)
    s = w_exec.sum()
    if s > eps:
        w_exec = w_exec / s
    else:
        # fallback: alles Cash auf erstes Asset (sollte praktisch nie passieren)
        w_exec = np.zeros_like(w_exec)
        w_exec[0] = 1.0

    return w_exec, l1_req, scale

def apply_execution_controls(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    delta_l1: float = 0.03,
    tau_l1: float = 0.20,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Pipeline: zuerst No-Trade-Band, dann Turnover-Cap.
    Gibt finale Gewichte + Info-Dict zurück.
    """
    w_after_ntb, l1_dist = no_trade_band(w_prev, w_target, delta_l1=delta_l1)
    if np.allclose(w_after_ntb, w_prev):
        return w_prev, {"acted": 0.0, "l1_dist": l1_dist, "scale": 0.0, "l1_step": 0.0}
    w_exec, l1_req, scale = apply_turnover_cap(w_prev, w_after_ntb, tau_l1=tau_l1)
    l1_step = float(np.abs(w_exec - w_prev).sum())
    return w_exec, {"acted": 1.0, "l1_dist": l1_dist, "l1_req": l1_req, "scale": scale, "l1_step": l1_step}