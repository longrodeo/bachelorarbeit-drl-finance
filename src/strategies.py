# src/strategies.py
import pandas as pd
import numpy as np
from src.returns import Returns
from src.kpis import KPI

class Strategies:
    """
    Sammlung verschiedener Portfolio-Strategien:
    - SMA-Crossover-Signale pro Asset
    - Risikoadjustierte Gewichtungen basierend auf Sharpe, Sortino, Calmar
    - Optimierung nach Treynor und Jensen's Alpha (marktbezogen)
    - Optimierung für minimale Beta und minimale Korrelation (marktunkorreliert)
    """

    @staticmethod
    def sma_crossover(prices: pd.DataFrame,
                      fast: int = 50,
                      slow: int = 200) -> pd.DataFrame:
        """
        Generiert pro Asset 0/1-Signale für SMA-Crossover:
        1 = Long wenn SMA_fast > SMA_slow, sonst 0 (Cash)

        Args:
            prices: DataFrame mit Preisen (Index: Datum, Spalten: Assets)
            fast: Fensterlänge für schnelle SMA
            slow: Fensterlänge für langsame SMA

        Returns:
            DataFrame mit 0/1-Signalen je Asset
        """
        sma_fast = prices.rolling(fast).mean()
        sma_slow = prices.rolling(slow).mean()
        signals = (sma_fast > sma_slow).astype(int)
        return signals

    @staticmethod
    def risk_adjusted_weights(prices: pd.DataFrame,
                               method: str = 'sharpe',
                               max_weight: float = 1.0) -> pd.DataFrame:
        """
        Berechnet globale Gewichtungen je Asset basierend auf Kennzahlen:
        wähle 'sharpe', 'sortino' oder 'calmar'.

        Args:
            prices: DataFrame mit Preisen
            method: Kennzahlname ('sharpe', 'sortino', 'calmar')
            max_weight: Maximales Gewicht je Asset

        Returns:
            DataFrame mit einem Zeitschritt (Enddatum) und optimalen Gewichten
        """
        log_ret = Returns.log(prices)
        scores = {}
        for asset in log_ret.columns:
            series = log_ret[asset]
            if method == 'sharpe':
                scores[asset] = KPI.sharpe_ratio(series, 0.0)
            elif method == 'sortino':
                scores[asset] = KPI.sortino_ratio(series, 0.0)
            elif method == 'calmar':
                scores[asset] = KPI.calmar_ratio(series)
            else:
                raise ValueError(f"Unknown method '{method}'")
        score_df = pd.DataFrame(scores, index=[prices.index[-1]])
        # negative Scores entfernen
        score_df = score_df.clip(lower=0)
        # Normieren auf Summe=1
        weights = score_df.div(score_df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        # Beschränkung und Renormalisierung
        def cap_row(row):
            capped = row.clip(upper=max_weight)
            total = capped.sum()
            return capped/total if total>0 else row
        weights = weights.apply(cap_row, axis=1)
        return weights

    @staticmethod
    def optimize_treynor(prices: pd.DataFrame,
                         market_returns: pd.Series,
                         risk_free_rate: pd.Series) -> pd.DataFrame:
        """
        Findet statische Gewichte (2 Assets) zur Maximierung der Treynor Ratio.

        Args:
            prices: DataFrame mit 2 Assets
            market_returns: Markt-Log-Returns
            risk_free_rate: Serie mit täglichen RF-Raten

        Returns:
            DataFrame mit einer Zeile: optimales Gewicht in Asset0 und Asset1
        """
        log_ret = Returns.log(prices)
        assets = log_ret.columns.tolist()
        best = {'weight':None, 'ratio':-np.inf}
        for w in np.linspace(0,1,101):
            port = w*log_ret[assets[0]] + (1-w)*log_ret[assets[1]]
            beta = KPI.estimate_beta(port, market_returns)
            tr = KPI.treynor_ratio(port, beta, risk_free_rate.mean())
            if tr>best['ratio']:
                best = {'weight':w, 'ratio':tr}
        return pd.DataFrame([{assets[0]: best['weight'], assets[1]: 1-best['weight']}], index=[prices.index[-1]])

    @staticmethod
    def optimize_jensen(prices: pd.DataFrame,
                         market_returns: pd.Series,
                         risk_free_rate: pd.Series) -> pd.DataFrame:
        """
        Findet statische Gewichte (2 Assets) zur Maximierung von Jensen's Alpha.
        """
        log_ret = Returns.log(prices)
        assets = log_ret.columns.tolist()
        best = {'weight':None, 'alpha':-np.inf}
        rm = KPI.annualized_return(market_returns)
        rf = risk_free_rate.mean()
        for w in np.linspace(0,1,101):
            port = w*log_ret[assets[0]] + (1-w)*log_ret[assets[1]]
            beta = KPI.estimate_beta(port, market_returns)
            alpha = KPI.jensen_alpha(port, market_returns, beta, rf)
            if alpha>best['alpha']:
                best = {'weight':w, 'alpha':alpha}
        return pd.DataFrame([{assets[0]: best['weight'], assets[1]: 1-best['weight']}], index=[prices.index[-1]])

    @staticmethod
    def optimize_min_beta(prices: pd.DataFrame,
                           market_returns: pd.Series) -> pd.DataFrame:
        """
        Findet statische Gewichte (2 Assets) die Beta zum Markt minimieren.
        """
        log_ret = Returns.log(prices)
        assets = log_ret.columns.tolist()
        best = {'weight':None, 'beta':np.inf}
        for w in np.linspace(0,1,101):
            port = w*log_ret[assets[0]] + (1-w)*log_ret[assets[1]]
            beta = KPI.estimate_beta(port, market_returns)
            if abs(beta)<best['beta']:
                best = {'weight':w, 'beta':abs(beta)}
        return pd.DataFrame([{assets[0]: best['weight'], assets[1]: 1-best['weight']}], index=[prices.index[-1]])

    @staticmethod
    def optimize_min_corr(prices: pd.DataFrame,
                           market_returns: pd.Series) -> pd.DataFrame:
        """
        Findet statische Gewichte (2 Assets) die Korrelation mit dem Markt minimieren.
        """
        log_ret = Returns.log(prices)
        assets = log_ret.columns.tolist()
        best = {'weight':None, 'corr':np.inf}
        for w in np.linspace(0,1,101):
            port = w*log_ret[assets[0]] + (1-w)*log_ret[assets[1]]
            corr = log_ret.corrwith(market_returns).iloc[0] if False else port.corr(market_returns)
            if abs(corr)<best['corr']:
                best = {'weight':w, 'corr':abs(corr)}
        return pd.DataFrame([{assets[0]: best['weight'], assets[1]: 1-best['weight']}], index=[prices.index[-1]])
