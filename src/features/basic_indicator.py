from __future__ import annotations
import pandas as pd
import numpy as np

# In diesem Modul werden Basis Indikatoren berechnet welche dem Modul im State 0 übergeben werden
# Der High, Low, Open und Close Preis sowie das Volumen werden direkt aus der Datenbasis übergeben


# ------------------------- logarithmische Returns und lineare Returns -------------------------

def returns(close: pd.Series, kind: str = "log") -> pd.Series:
    if kind == "log":
        return np.log(close / close.shift(1))
    return close.pct_change()

# ------------------------- Spread und Vola Schätzung -------------------------

def corwin_schultz_beta(high: pd.Series, low: pd.Series, sample_length: int = 1) -> pd.Series:
    hl = np.log(high / low) ** 2
    beta = hl.rolling(2).sum()
    if sample_length and sample_length > 1:
        beta = beta.rolling(sample_length).mean()
    return beta

def corwin_schultz_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    h_max = high.rolling(2).max()
    l_min = low.rolling(2).min()
    return np.log(h_max / l_min) ** 2

def corwin_schultz_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(2.0) - 1.0) / den) * np.sqrt(beta)
    alpha = alpha - np.sqrt(gamma / den)
    return alpha.clip(lower=0.0)

def corwin_schultz_spread(alpha: pd.Series) -> pd.Series:
    ex_alpha = np.exp(alpha)
    return 2.0 * (ex_alpha - 1.0) / (1.0 + ex_alpha)

def becker_parkinson_sigma(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    k2 = np.sqrt(8.0 / np.pi)
    den = 3.0 - 2.0 * np.sqrt(2.0)
    sigma = (2.0 ** -0.5 - 1.0) * (np.sqrt(beta) / (k2 * den))
    sigma = sigma + np.sqrt(gamma / (k2 ** 2 * den))
    return sigma.clip(lower=0.0)


