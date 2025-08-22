from __future__ import annotations
import numpy as np
import pandas as pd


def _safe_rolling(s: pd.Series, window: int, min_periods: int | None = None):
    if min_periods is None:
        min_periods = window
    return s.rolling(window=window, min_periods=min_periods)


# ------------------------- technische Indikatoren -------------------------
# In State 1 werden zu State 0 verschiedene technische Indikatoren hinzugezogen zur Bestimmung
# von Liquidität des Marktes sowie Trends

def average_dollar_volume(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    return (close * volume).rolling(window).mean()


def simple_moving_average(s: pd.Series, window: int) -> pd.Series:
    return _safe_rolling(s, window).mean()


def exponential_moving_average(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def relative_strength_index(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def moving_average_convergence_divergence(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = exponential_moving_average(close, fast)
    ema_slow = exponential_moving_average(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = exponential_moving_average(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = simple_moving_average(close, window)
    std = _safe_rolling(close, window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=False
    )
    denom = 0.015 * mad.replace(0, np.nan)
    cci_val = (tp - sma_tp) / denom
    cci_val.name = f"cci_{period}"
    return cci_val

def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movements
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Wilder smoothing via EMA(alpha=1/period)
    alpha = 1.0 / period
    tr_sm = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # DIs
    eps = 1e-12
    plus_di = 100.0 * (plus_dm_sm / (tr_sm.replace(0, np.nan)))
    minus_di = 100.0 * (minus_dm_sm / (tr_sm.replace(0, np.nan)))

    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    out = pd.DataFrame({
        f"adx_{period}": adx_val,
        f"plus_di_{period}": plus_di,
        f"minus_di_{period}": minus_di,
    }, index=close.index)
    return out

DEFAULTS = {
    "sma": [20, 60],
    "rsi": 14,
    "macd": (12, 26, 9),
    "boll": (20, 2.0),
    "adv": 20,
    "cci": 20
}

