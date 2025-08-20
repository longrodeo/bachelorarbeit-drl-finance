import pandas as pd
from src.data.calendar import nyse_trading_days
from src.data.align import align_to_trading_days, resample_crypto_last

def test_align_basic():
    cal = nyse_trading_days(start="2024-01-01", end="2024-01-15")
    # ETF-Reihe: nur 2024-01-02 & 2024-01-05 vorhanden
    etf = pd.DataFrame({"Adj Close":[100,101]},
                       index=pd.to_datetime(["2024-01-02","2024-01-05"], utc=True))
    out = align_to_trading_days(etf, cal)
    assert out.index.equals(cal) and out.loc["2024-01-02"].notna().all()
    # Crypto: tägliche Serie + Wochenende → last auf Handelstage
    crypto = pd.DataFrame({"close":[1,2,3,4,5,6,7]},
                          index=pd.to_datetime(["2023-12-31","2024-01-01","2024-01-02",
                                               "2024-01-03","2024-01-04","2024-01-06","2024-01-07"], utc=True))
    out2 = resample_crypto_last(crypto, cal)
    assert out2.index.equals(cal)
