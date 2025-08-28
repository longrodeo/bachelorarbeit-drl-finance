# ---------------------------------------------------------------------------
# Datei: src/features/basic_indicator.py
# Zweck: Berechnung elementarer Preis- und Volumenindikatoren.
# Hauptfunktionen: ``returns`` sowie Spread- und Volaschätzer nach
#   Corwin/Schultz bzw. Becker/Parkinson.
# Ein-/Ausgabe: ``pd.Series`` mit Hoch/Tief/Schlusskursen und Volumen.
# Abhängigkeiten: ``pandas``, ``numpy``; wichtig sind vollständige, sortierte
#   Zeitreihen ohne fehlende Werte.
# ---------------------------------------------------------------------------

"""Basisindikatoren auf Preis- und Volumendaten.
Enthält Renditeberechnungen sowie Schätzer für Spread und Volatilität nach
Corwin/Schultz und Becker/Parkinson. Einsatz im frühen State der
Feature-Pipeline."""

from __future__ import annotations  # zukunftsfähige Typannotationen (Python 3.7+)
import pandas as pd  # DataFrame/Series als zentrale Datenstrukturen
import numpy as np  # effiziente numerische Routinen

# Hinweis: High/Low/Open/Close sowie Volumen stammen direkt aus dem Preis-Feed.


# ------------------------- logarithmische Returns und lineare Returns -------------------------

def returns(close: pd.Series, kind: str = "log") -> pd.Series:
    """Logarithmische oder lineare Rendite berechnen.

    Parameters
    ----------
    close : pd.Series
        Schlusskurse mit DatetimeIndex.
    kind : str, optional
        "log" für ``ln(P_t/P_{t-1})`` oder alles andere für lineare Renditen.

    Returns
    -------
    pd.Series
        Renditeserie mit ``NaN`` an der ersten Stelle.
    """
    if kind == "log":  # Zweig für logarithmische Rendite
        return np.log(close / close.shift(1))  # ln(P_t / P_{t-1})
    return close.pct_change()  # lineare prozentuale Veränderung

# ------------------------- Spread und Vola Schätzung -------------------------

def corwin_schultz_beta(high: pd.Series, low: pd.Series, sample_length: int = 1) -> pd.Series:
    """Beta-Term der Corwin-Schultz-Spread-Schätzung.

    Parameters
    ----------
    high, low : pd.Series
        Tageshochs und -tiefs.
    sample_length : int, optional
        Fenster für optionales Nachglätten (Standard: 1 = kein Glätten).

    Returns
    -------
    pd.Series
        Gleitend berechnetes ``beta``.
    """
    hl = np.log(high / low) ** 2  # quadrierte Intraday-Range
    beta = hl.rolling(2).sum()  # Summation der letzten zwei Tage
    if sample_length and sample_length > 1:  # optionales Glätten über mehrere Tage
        beta = beta.rolling(sample_length).mean()  # Mittelwert über Fenster
    return beta  # liefert Serie mit Beta-Werten

def corwin_schultz_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """Gamma-Term: zweiteilige Hoch/Tief-Spanne.

    Parameters
    ----------
    high, low : pd.Series
        Hoch- und Tiefkurse der Tage.

    Returns
    -------
    pd.Series
        ``gamma``-Werte als Basis für Spread/Volatilität.
    """
    h_max = high.rolling(2).max()  # Maximum der letzten zwei Hochs
    l_min = low.rolling(2).min()  # Minimum der letzten zwei Tiefs
    return np.log(h_max / l_min) ** 2  # logarithmische Spannweite im Quadrat

def corwin_schultz_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """Alpha-Term: Kombination aus Beta und Gamma.

    Parameters
    ----------
    beta, gamma : pd.Series
        Vorberechnete Beta- und Gamma-Terme.

    Returns
    -------
    pd.Series
        ``alpha``-Serie, negative Werte werden auf 0 gesetzt.
    """
    den = 3.0 - 2.0 * np.sqrt(2.0)  # Nenner der CS-Formel
    alpha = ((np.sqrt(2.0) - 1.0) / den) * np.sqrt(beta)  # erster Summand
    alpha = alpha - np.sqrt(gamma / den)  # zweiter Summand
    return alpha.clip(lower=0.0)  # Spread ist nicht negativ definierbar

def corwin_schultz_spread(alpha: pd.Series) -> pd.Series:
    """Bid-Ask-Spread aus dem Alpha-Term ableiten.

    Parameters
    ----------
    alpha : pd.Series
        Ergebnis von ``corwin_schultz_alpha``.

    Returns
    -------
    pd.Series
        Geschätzter relativer Spread.
    """
    ex_alpha = np.exp(alpha)  # e^{alpha}
    return 2.0 * (ex_alpha - 1.0) / (1.0 + ex_alpha)  # Formel aus CS-Paper

def becker_parkinson_sigma(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """Volatilitätsabschätzung nach Becker/Parkinson.

    Parameters
    ----------
    beta, gamma : pd.Series
        Vorher berechnete CS-Terme.

    Returns
    -------
    pd.Series
        Nicht-negative Volatilitätsabschätzung ``sigma``.
    """
    k2 = np.sqrt(8.0 / np.pi)  # Konstante aus der Herleitung
    den = 3.0 - 2.0 * np.sqrt(2.0)  # gemeinsamer Nenner
    sigma = (2.0 ** -0.5 - 1.0) * (np.sqrt(beta) / (k2 * den))  # erster Summand
    sigma = sigma + np.sqrt(gamma / (k2 ** 2 * den))  # zweiter Summand
    return sigma.clip(lower=0.0)  # Volatilität darf nicht negativ sein


