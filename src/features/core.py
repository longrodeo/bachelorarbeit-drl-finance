import numpy as np
import pandas as pd

def returns(close: pd.Series, kind: str = "log") -> pd.Series:
    if kind == "log":
        return np.log(close / close.shift(1))
    return close.pct_change()

def adv(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    return (close * volume).rolling(window).mean()

def parkinson_sigma(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hl_var = (np.log(high / low)) ** 2
    return (hl_var / (4.0 * np.log(2.0))).rolling(window).mean()
