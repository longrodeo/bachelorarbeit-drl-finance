from typing import Tuple
import numpy as np

def no_trade_band(
    weights_prev: np.ndarray,
    weights_target: np.ndarray,
    min_change_l1: float = 0.03
) -> Tuple[np.ndarray, float]:
    """
    Rebalancing erst ab einer spürbaren Zieländerung.
    Kriterium: L1-Abstand (sum(abs(target - prev))) > min_change_l1.

    Args:
        weights_prev: Aktuelle Gewichte (Summe≈1, long-only erwartet)
        weights_target: Zielgewichte (Summe≈1, long-only erwartet)
        min_change_l1: L1-Schwelle (z. B. 0.03 == 3 % der "Torte")

    Returns:
        weights_exec : Entweder unverändert (prev) oder Ziel (target)
        l1_distance  : Gemessener L1-Abstand zwischen prev & target
    """
    weights_prev = np.asarray(weights_prev, dtype=float)
    weights_target = np.asarray(weights_target, dtype=float)
    if weights_prev.shape != weights_target.shape:
        raise ValueError("weights_prev und weights_target müssen die gleiche Form haben")

    l1_distance = float(np.abs(weights_target - weights_prev).sum())
    if l1_distance > min_change_l1:
        return weights_target, l1_distance  # handeln
    else:
        return weights_prev, l1_distance    # nichts tun


def apply_turnover_cap(
    weights_prev: np.ndarray,
    weights_target: np.ndarray,
    max_step_l1: float = 0.30,
    eps: float = 1e-12
) -> Tuple[np.ndarray, float, float]:
    """
    Begrenze den Rebalance-Schritt (L1-Norm) auf max_step_l1.

    - Wenn L1(target - prev) <= max_step_l1: voll zum Ziel (applied_scale=1).
    - Sonst: nur anteilig in Richtung Ziel mit applied_scale = max_step_l1 / L1(...).

    Hinweise:
    - Liegen prev & target auf dem Simplex (>=0, Summe=1), bleibt das Ergebnis das auch
      (Konvexkombination). Kleine numerische Abweichungen werden renormalisiert.

    Returns:
        weights_exec   : Ausgeführte Gewichte nach Cap
        l1_requested   : Ursprünglicher L1-Abstand target - prev
        applied_scale  : Verwendeter Schrittfaktor in [0,1]
    """
    weights_prev = np.asarray(weights_prev, dtype=float)
    weights_target = np.asarray(weights_target, dtype=float)
    if weights_prev.shape != weights_target.shape:
        raise ValueError("weights_prev und weights_target müssen die gleiche Form haben")
    if max_step_l1 < 0:
        raise ValueError("max_step_l1 muss >= 0 sein")

    diff = weights_target - weights_prev
    l1_requested = float(np.abs(diff).sum())

    if l1_requested <= max_step_l1 + eps or l1_requested <= eps or max_step_l1 >= 2.0 - eps:
        weights_exec = weights_target.copy()
        applied_scale = 1.0
    else:
        applied_scale = float(max_step_l1 / l1_requested)
        weights_exec = weights_prev + applied_scale * diff

    # numerische Sanity: Simplex sichern
    weights_exec = np.maximum(weights_exec, 0.0)

    # Budget-Pfad einhalten (nur nach unten begrenzen, kein Hochskalieren -> Cash bleibt Cash)
    s_prev = float(weights_prev.sum())
    s_tgt  = float(weights_target.sum())
    s_desired = s_prev + applied_scale * (s_tgt - s_prev)

    s_cur = float(weights_exec.sum())
    if s_cur > 0.0:
        scale = min(1.0, s_desired / s_cur)  # nur downscalen
        if scale < 1.0 - 1e-12:
            weights_exec *= scale

    return weights_exec, l1_requested, applied_scale


def apply_execution_controls(
    weights_prev: np.ndarray,
    weights_target: np.ndarray,
    min_change_l1: float = 0.03,
    max_step_l1: float = 0.20,
) -> Tuple[np.ndarray, dict[str, float]]:
    """
    Pipeline: zuerst No-Trade-Band, dann Turnover-Cap.
    Gibt finale Gewichte + Info-Dict zurück.
    """
    after_ntb, l1_distance = no_trade_band(
        weights_prev, weights_target, min_change_l1=min_change_l1
    )
    if np.allclose(after_ntb, weights_prev):
        return weights_prev, {"acted": 0.0, "l1_distance": l1_distance, "applied_scale": 0.0, "l1_step": 0.0}

    weights_exec, l1_requested, applied_scale = apply_turnover_cap(
        weights_prev, after_ntb, max_step_l1=max_step_l1
    )
    l1_step = float(np.abs(weights_exec - weights_prev).sum())
    return weights_exec, {
        "acted": 1.0,
        "l1_distance": l1_distance,
        "l1_requested": l1_requested,
        "applied_scale": applied_scale,
        "l1_step": l1_step,
    }
