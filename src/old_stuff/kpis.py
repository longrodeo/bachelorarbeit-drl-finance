# src/kpis.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import skew
from scipy.stats import kurtosis

class KPI:
    """
    Sammlung von Kennzahlen zur Bewertung von Portfolios.
    Alle Methoden erwarten einen DataFrame oder Series mit log returns.
    """

    @staticmethod
    def annualized_return(log_returns: pd.Series) -> float:
        """Gibt die annualisierte Durchschnittsrendite zurück."""
        return log_returns.mean() * 252

    @staticmethod
    def annualized_volatility(log_returns: pd.Series) -> float:
        """Berechnet die annualisierte Volatilität (Standardabweichung)."""
        return log_returns.std() * np.sqrt(252)

    @staticmethod
    def sharpe_ratio(log_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Sharpe Ratio: Risk-adjusted return."""
        excess_returns = log_returns - risk_free_rate / 252
        return (excess_returns.mean() / log_returns.std()) * np.sqrt(252)

    @staticmethod
    def sortino_ratio(log_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Sortino Ratio: Wie Sharpe, aber nur downside risk berücksichtigt."""
        excess_returns = log_returns - risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        downside_std = negative_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else np.nan

    @staticmethod
    def max_drawdown(cumulative_returns: pd.Series) -> float:
        """Größter kumulierter Verlust vom Hoch zum Tief."""
        peak = cumulative_returns.cummax()
        drawdown = cumulative_returns / peak - 1
        return drawdown.min()


    @staticmethod
    def calmar_ratio(log_returns: pd.Series) -> float:
        """Calmar Ratio = annualized return / max drawdown."""
        cumulative_returns = np.exp(log_returns.cumsum())
        max_dd = KPI.max_drawdown(cumulative_returns)
        ann_return = KPI.annualized_return(log_returns)
        return ann_return / abs(max_dd) if max_dd != 0 else np.nan


    @staticmethod
    def estimate_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """Schätzt Beta über lineare Regression."""
        cov = np.cov(portfolio_returns, market_returns)
        return cov[0,1] / cov[1,1]  # Cov(Rp, Rm) / Var(Rm)
        
    
    @staticmethod
    def jensen_alpha(log_returns: pd.Series, market_returns: pd.Series, beta: float,
                 risk_free_rate: float = 0.0) -> float:
        """Jensen's Alpha: Outperformance im Verhältnis zum Markt."""
        rp = KPI.annualized_return(log_returns)
        rm = KPI.annualized_return(market_returns)
        return rp - (risk_free_rate + beta * (rm - risk_free_rate))


    @staticmethod
    def treynor_ratio(log_returns: pd.Series, beta: float, risk_free_rate: float = 0.0) -> float:
        """Treynor Ratio: Risk-adjusted return mit systematischem Risiko (Beta)."""
        rp = KPI.annualized_return(log_returns)
        return (rp - risk_free_rate) / beta if beta != 0 else np.nan


    @staticmethod
    def value_at_risk(log_returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Value at Risk (VaR) mit Varianz-Kovarianz-Methode."""
        mean = log_returns.mean()
        std = log_returns.std()
        z = norm.ppf(1 - confidence_level)
        return -(z * std - mean)  # negativ = Verlustbetrag
        

    @staticmethod
    def skewness(log_returns: pd.Series) -> float:
        """Asymmetrie der Verteilung."""
        return skew(log_returns)
        

    @staticmethod
    def kurtosis(log_returns: pd.Series) -> float:
        """Fat tails: Schiefe Schwankungsverteilung."""
        return kurtosis(log_returns, fisher=False)  # Fisher=False → normal = 3


    @staticmethod
    def get_kpi_report(log_returns: pd.Series,
                   market_returns: pd.Series = None,
                   risk_free_rate: float = 0.0,
                   beta: float = None,
                   confidence_level: float = 0.95) -> dict:
        """Berechnet alle verfügbaren KPIs auf einmal."""
        report = {
            "Ann. Return": KPI.annualized_return(log_returns),
            "Ann. Volatility": KPI.annualized_volatility(log_returns),
            "Sharpe Ratio": KPI.sharpe_ratio(log_returns, risk_free_rate),
            "Sortino Ratio": KPI.sortino_ratio(log_returns, risk_free_rate),
            "Max Drawdown": KPI.max_drawdown(np.exp(log_returns.cumsum())),
            "Calmar Ratio": KPI.calmar_ratio(log_returns),
            "Value at Risk": KPI.value_at_risk(log_returns, confidence_level),
            "Skewness": KPI.skewness(log_returns),
            "Kurtosis": KPI.kurtosis(log_returns)
        }
    
        if market_returns is not None and beta is not None:
            report["Treynor Ratio"] = KPI.treynor_ratio(log_returns, beta, risk_free_rate)
            report["Jensen Alpha"] = KPI.jensen_alpha(log_returns, market_returns, beta, risk_free_rate)
    
        return report

