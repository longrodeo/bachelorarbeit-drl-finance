import numpy as np

def action_to_weights_softmax(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map beliebige R^N-Aktionen auf long-only Gewichte mit Summe=1."""
    a = np.asarray(a, dtype=float)
    a = a - np.max(a)      # numerisch stabil für exp
    e = np.exp(a)
    s = e.sum()
    return e / max(s, eps)
