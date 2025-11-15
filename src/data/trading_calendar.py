# -----------------------------------------------------------------------------
# Provides NYSE trading day indices used across the data pipeline, falling back
# to simple business-day ranges when exchange calendars are unavailable.
# -----------------------------------------------------------------------------

"""Generate NYSE trading session indices for aligning financial datasets."""

from datetime import datetime
import pandas as pd

try:
    import exchange_calendars as xcals
    _CAL_LIB = "exchange_calendars"
except ImportError:
    _CAL_LIB = None


def nyse_trading_days(start="2000-01-01", end=None, tz="UTC") -> pd.DatetimeIndex:
    """Return NYSE trading sessions between ``start`` and ``end``.

    Parameters
    ----------
    start : str
        Inclusive start date in ISO format.
    end : str | None
        Inclusive end date; defaults to today when ``None``.
    tz : str
        Target timezone for the resulting index.

    Returns
    -------
    pd.DatetimeIndex
        Sorted timezone-aware trading sessions.
    """
    end = end or datetime.utcnow().date().isoformat()
    if _CAL_LIB == "exchange_calendars":
        cal = xcals.get_calendar("XNYS")
        sched = cal.schedule.loc[start:end]
        days = pd.DatetimeIndex(
            sched.index.tz_localize("America/New_York").tz_convert(tz).normalize()
        )
        return days.unique().sort_values()
    else:
        return pd.date_range(start=start, end=end, freq="B", tz=tz)


if __name__ == "__main__":
    idx = nyse_trading_days(start="2019-01-01")
    print("Number of trading days since 2019-01-01:", len(idx))
    print("First 5:", idx[:5].tolist())
