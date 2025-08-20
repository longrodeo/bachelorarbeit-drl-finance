import pandas as pd
from src.data.calendar import nyse_trading_days

def test_nyse_trading_days_basic():
    idx = nyse_trading_days(start="2024-01-01", end="2024-01-31")
    # 1) Typ & Sortierung
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.is_monotonic_increasing
    # 2) Zeitzone (UTC-aware)
    assert idx.tz is not None and str(idx.tz) == "UTC"
    # 3) Kein Wochenende
    assert all(ts.weekday() < 5 for ts in idx)
    # 4) Kein leerer Index
    assert len(idx) > 0
