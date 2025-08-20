# src/data/calendar.py
from datetime import datetime
import pandas as pd

try:
    import exchange_calendars as xcals  # für echten NYSE-Kalender
    _CAL_LIB = "exchange_calendars"
except ImportError:
    _CAL_LIB = None

def nyse_trading_days(start="2000-01-01", end=None, tz="UTC") -> pd.DatetimeIndex:
    """
    Liefert NYSE-Handelstage als DatetimeIndex (tz-aware).
    - Schließt Wochenenden & NYSE-Feiertage aus.
    """
    end = end or datetime.utcnow().date().isoformat()
    if _CAL_LIB == "exchange_calendars":
        cal = xcals.get_calendar("XNYS")
        # schedule ist ein DataFrame mit market_open & market_close
        sched = cal.schedule.loc[start:end]
        # Index enthält die Handelstage → auf gewünschte tz normieren
        days = pd.DatetimeIndex(
            sched.index.tz_localize("America/New_York").tz_convert(tz).normalize()
        )
        return days.unique().sort_values()
    else:
        # Fallback: nur Werktage (ohne Feiertage)
        return pd.date_range(start=start, end=end, freq="B", tz=tz)

if __name__ == "__main__":
    idx = nyse_trading_days(start="2019-01-01")
    print("Anzahl Handelstage seit 2019-01-01:", len(idx))
    print("Erste 5:", idx[:5].tolist())
