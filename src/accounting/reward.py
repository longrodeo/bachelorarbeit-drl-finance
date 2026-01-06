# src/accounting/reward.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class RewardSpec:
    """
    Definition der Reward-Variante und Parameter.

    kind:
      - "log"                 : r_t = log(NAV_{t+1}/NAV_t)
      - "icvar"               : r_t = log(...) - lambda_ * ICVaR_t
      - "icvar_dd"            : r_t = log(...) - lambda_ * ICVaR_t - gamma * ΔMDD_t

    icvar_mode:
      - "ex_ante"             : CVaR_t aus Returns bis t-1 (empfohlen, klare Kausalität)
      - "ex_post"             : CVaR_t aus Returns bis t   (Ablation/Variante)

    estimator:
      - "rolling"             : CVaR mit Rolling-Fenster; optional Nachglättung via ewm_alpha
    """
    kind: Literal["log", "icvar", "icvar_dd"] = "log"
    icvar_mode: Literal["ex_ante", "ex_post"] = "ex_ante"
    alpha: float = 0.05               # Tail-Level für (C)VaR
    min_period: int = 20                 # Expanding-Fenster für Training höher setzten
    lambda_: float = 25.0              # Gewicht ICVaR
    gamma: float = 1.0                # Gewicht ΔMDD (nur bei kind="icvar_dd")
    estimator: Literal["rolling"] = "rolling"
    ewm_alpha: float | None = 0.10    # optional: Glättung der CVaR-Serie (0<alpha<=1), None = aus

def apply_reward_spec(*, r_log_t, icvar_t=0.0, delta_mdd_t=0.0, spec: RewardSpec):
    """
    Single source of truth.
    Funktioniert mit float, numpy arrays und pandas Series (broadcasting).
    """
    if spec.kind == "log":
        return r_log_t
    if spec.kind == "icvar":
        lamda = spec.lambda_
        # print(f"Lamda: {lamda}")
        return r_log_t - spec.lambda_ * icvar_t
    if spec.kind == "icvar_dd":
        return r_log_t - spec.lambda_ * icvar_t - spec.gamma * delta_mdd_t, print(spec.gamma)
    raise ValueError(f"Unbekannte Reward-Variante: {spec.kind}")