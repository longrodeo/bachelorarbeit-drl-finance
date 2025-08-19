# src/data_loader.py

import yfinance as yf
import pandas as pd


def load_prices(ticker_list, start: str, end: str) -> pd.DataFrame:
    """
    Lädt historische Schlusskurse (Close) für mehrere Ticker.
    Empfohlen: direkt TR-Ticker wie ^SP500TR nutzen.
    """
    data = yf.download(ticker_list, start=start, end=end)["Close"]
    return data.dropna()

