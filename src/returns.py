# src/returns.py

import pandas as pd
import numpy as np

class Returns:
    """Hilfsklasse zur Berechnung von Renditen aus Preiszeitreihen."""

    @staticmethod
    def log(prices: pd.DataFrame) -> pd.DataFrame:
        """Logarithmische Rendite: ln(Pt / Pt-1)"""
        return np.log(prices / prices.shift(1)).dropna()
        

    @staticmethod
    def linear(prices: pd.DataFrame) -> pd.DataFrame:
        """Einfache Rendite: (Pt - Pt-1) / Pt-1"""
        return prices.pct_change().dropna()
