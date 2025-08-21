import numpy as np
import pandas as pd

def cs_beta(high: pd.Series, low: pd.Series, sl: int = 1) -> pd.Series:
    hl = np.log(high / low) ** 2
    beta = hl.rolling(2).sum()
    if sl and sl > 1:
        beta = beta.rolling(sl).mean()
    return beta

def cs_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    return np.log(h2 / l2) ** 2

def cs_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    den = 3.0 - 2.0 * np.sqrt(2.0)
    term1 = ((np.sqrt(2.0) - 1.0) / den) * np.sqrt(beta)
    term2 = np.sqrt(gamma / den)
    return (term1 - term2).clip(lower=0.0)

def cs_sigma(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    k2 = np.sqrt(8.0 / np.pi)
    den = 3.0 - 2.0 * np.sqrt(2.0)
    term1 = (2.0 ** -0.5 - 1.0) * (np.sqrt(beta) / (k2 * den))
    term2 = np.sqrt(gamma / (k2 ** 2 * den))
    return (term1 + term2).clip(lower=0.0)

def cs_spread_from_alpha(alpha: pd.Series) -> pd.Series:
    ealpha = np.exp(alpha)
    return 2.0 * (ealpha - 1.0) / (1.0 + ealpha)
