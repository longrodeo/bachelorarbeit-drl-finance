# -*- coding: utf-8 -*-
# src/rl/wrappers.py
#
# Gymnasium-Wrapper, der Agent-Aktionen (Logits in R^{A+1}) via Softmax
# in long-only Gewichte inkl. Cash (Summe=1) mapped und an deine Env weitergibt.

from __future__ import annotations
import gymnasium as gym
import numpy as np

# Falls dein Mapping bereits in src/policy/action_mapping.py liegt, importiere es von dort:
try:
    from src.policy.action_mapping import action_to_weights_softmax  # deine Version
except Exception:
    # Fallback: direkte Kopie der Funktion (wie von dir geliefert)
    def action_to_weights_softmax(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        a = a - np.max(a)
        e = np.exp(a)
        s = e.sum()
        return e / max(s, eps)

class ActionMappingWrapper(gym.ActionWrapper):
    """
    Erwartet vom Agenten *Logits* (beliebige reelle Zahlen) der Länge A+1.
    Mappt sie per Softmax in Gewichte (>=0, Summe=1, letzter Eintrag = Cash) und
    ruft damit env.step(mapped).
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Wir lassen den Agenten auf einem "weiten" Aktionsraum arbeiten:
        A_plus_1 = env.action_space.shape[0]
        # Unbeschränkter Raum kann manche Algos stören; [-5, 5] ist praktisch groß.
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(A_plus_1,), dtype=np.float32)

    def action(self, act_logits):
        act_logits = np.asarray(act_logits, dtype=np.float64)
        w = action_to_weights_softmax(act_logits)
        # Optional: minimale Cash-Quote erzwingen (z.B. 5%)
        # if w[-1] < 0.05:
        #     deficit = 0.05 - w[-1]
        #     w[:-1] = np.clip(w[:-1] - deficit * w[:-1] / (w[:-1].sum() + 1e-12), 0.0, None)
        #     w = w / w.sum()
        return w.astype(np.float32)
