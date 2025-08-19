# src/riskfree.py

import pandas as pd
from pandas_datareader import data as web

class RiskFreeRate:
    @staticmethod
    def us_fred(date_index: pd.Index) -> pd.DataFrame:
        """
        Holt TB3MS von FRED und liefert täglichen Zinssatz (annual. / 252).
        Subtrahiert 1 Monat, damit der vorangehende Monatswert mitkommt.
        """
        # 1) Subtrahiere einen Monat statt nur 7 Tage
        start = date_index.min() - pd.DateOffset(months=1)
        end   = date_index.max()
        rf = web.DataReader("TB3MS", "fred", start, end) / 100

        # 2) Resample auf Business Days
        rf = rf.resample("B").ffill()

        # 3) Auf Dein Return-Index anpassen + fwd/bwd fill
        rf = rf.reindex(date_index).ffill().bfill()

        # 4) Auf daily (252) skalieren
        return rf["TB3MS"] / 252


    @staticmethod
    def eu_placeholder(date_index: pd.Index, rate: float = 0.0275) -> pd.DataFrame:
        """
        Gibt einen konstanten europäischen Zins (Platzhalter) zurück, daily skaliert.

        Parameters:
            date_index (pd.Index): Index mit Zeitstempeln
            rate (float): Jahreszins in Dezimalform (z. B. 0.0275 für 2,75 %)

        Returns:
            pd.DataFrame: täglicher Zinssatz
        """
        daily_rate = rate / 252
        rf = pd.Series(daily_rate, index=date_index, name="EU_Rate").to_frame()
        return rf
