# -----------------------------------------------------------------------------
# Alignment utilities that project raw price series onto a target trading
# calendar for both exchange-traded assets and 24/7 crypto markets.
# Used by the INTERIM pipeline stage prior to feature engineering.
# -----------------------------------------------------------------------------
"""Helper functions that align price data to a trading calendar."""

import pandas as pd  # Core library for time-series manipulation

def align_to_trading_days(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Align an equity time series to the provided trading sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Input series indexed by datetime.
    cal_idx : pd.DatetimeIndex
        Target trading calendar index (tz-aware).

    Returns
    -------
    pd.DataFrame
        Reindexed series where missing sessions remain ``NaN``.
    """
    if df.index.tz is None:  # ensure the index carries timezone information
        df = df.tz_localize("UTC")  # localize to UTC for consistent comparisons
    return df.reindex(cal_idx)  # reindex without filling gaps

def resample_crypto_last(df: pd.DataFrame, cal_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Condense a crypto time series to the provided trading sessions.

    The series is resampled to daily frequency first and then reindexed to
    the desired trading calendar to mimic exchange trading days.

    Parameters
    ----------
    df : pd.DataFrame
        High-frequency or daily time series indexed by datetime.
    cal_idx : pd.DatetimeIndex
        Target trading calendar index.

    Returns
    -------
    pd.DataFrame
        Series with one observation per trading session.
    """
    if df.index.tz is None:  # ensure the index is timezone-aware
        df = df.tz_localize("UTC")
    daily = df.resample("1D").last()  # aggregate to one value per calendar day
    return daily.reindex(cal_idx)  # project onto the trading sessions
