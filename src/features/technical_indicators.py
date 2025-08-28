# ---------------------------------------------------------------------------
# Datei: src/features/technical_indicators.py
# Zweck: Sammlung gängiger technischer Analysekennzahlen.
# Hauptfunktionen: gleitende Durchschnitte, RSI, MACD, Bollinger, CCI, ADX.
# Ein-/Ausgabe: Preis/Volumen-``pd.Series`` → Indikatoren als ``Series``/``DataFrame``.
# Abhängigkeiten: ``numpy``, ``pandas``; wichtig sind vollständige Indizes und
#   korrekte Fensterlängen.
# ---------------------------------------------------------------------------

"""Sammlung technischer Indikatoren für Preiszeitreihen.
Berechnet gleitende Durchschnitte, RSI, MACD, Bollinger-Bänder u. a.
Verwendung in späteren Pipeline-Stufen zur Trend- und Liquiditätsanalyse."""

from __future__ import annotations  # Nutzung von zukünftigen Typfeatures
import numpy as np  # numerische Routine (z. B. für Arrays)
import pandas as pd  # Series/DataFrame-Verarbeitung


def _safe_rolling(s: pd.Series, window: int, min_periods: int | None = None):
    """Rolling-Helper mit automatisch gesetztem ``min_periods``.

    Parameters
    ----------
    s : pd.Series
        Eingangsserie.
    window : int
        Fensterbreite.
    min_periods : int | None, optional
        Mindestanzahl Werte im Fenster; Default = ``window``.

    Returns
    -------
    Rolling
        Pandas-Rolling-Objekt zur weiteren Aggregation.
    """
    if min_periods is None:  # falls keine Mindestwerte angegeben
        min_periods = window  # Standard auf Fenstergröße setzen
    return s.rolling(window=window, min_periods=min_periods)  # Rolling-Objekt erzeugen


# ------------------------- technische Indikatoren -------------------------
# In State 1 werden verschiedene technische Indikatoren zur Trend- und
# Liquiditätsbeurteilung berechnet.

def average_dollar_volume(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Durchschnittlicher Handelswert (Preis×Volumen).

    Parameters
    ----------
    close, volume : pd.Series
        Schlusskurse und gehandeltes Volumen.
    window : int
        Länge des gleitenden Fensters.

    Returns
    -------
    pd.Series
        Durchschnittlicher Dollar-Umsatz je Tag.
    """
    return (close * volume).rolling(window).mean()  # Preis×Volumen → Mittelwert


def simple_moving_average(s: pd.Series, window: int) -> pd.Series:
    """Einfacher gleitender Durchschnitt.

    Parameters
    ----------
    s : pd.Series
        Eingangszeitreihe.
    window : int
        Fensterbreite.

    Returns
    -------
    pd.Series
        SMA-Werte.
    """
    return _safe_rolling(s, window).mean()  # Mittelwert über Fenster


def exponential_moving_average(s: pd.Series, span: int) -> pd.Series:
    """Exponentiell gewichteter Durchschnitt (EMA).

    Parameters
    ----------
    s : pd.Series
        Eingangsserie.
    span : int
        Glättungsspanne gemäß ``pd.Series.ewm``.

    Returns
    -------
    pd.Series
        EMA-Werte.
    """
    return s.ewm(span=span, adjust=False, min_periods=span).mean()  # EMA-Berechnung


def relative_strength_index(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI nach Wilder: misst Stärke von Auf‑ vs. Abwärtsbewegungen.

    Parameters
    ----------
    close : pd.Series
        Schlusskurse.
    period : int, optional
        Länge der Betrachtungsperiode (Standard 14 Tage).

    Returns
    -------
    pd.Series
        RSI-Werte zwischen 0 und 100.
    """
    delta = close.diff()  # Tagesänderungen
    up = delta.clip(lower=0.0)  # nur positive Bewegungen
    down = -delta.clip(upper=0.0)  # nur negative Bewegungen
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()  # geglättete Aufwärtsbewegung
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()  # geglättete Abwärtsbewegung
    rs = roll_up / roll_down.replace(0, np.nan)  # relative Stärke
    rsi = 100 - (100 / (1 + rs))  # RSI-Formel
    return rsi  # Serie zurückgeben


def moving_average_convergence_divergence(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD-Linie, Signal und Histogramm berechnen.

    Parameters
    ----------
    close : pd.Series
        Schlusskurse.
    fast, slow, signal : int, optional
        EMA-Perioden für schnelle/ langsame Linie und Signallinie.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        MACD-Linie, Signallinie und Histogramm.
    """
    ema_fast = exponential_moving_average(close, fast)  # schnelle EMA
    ema_slow = exponential_moving_average(close, slow)  # langsame EMA
    macd = ema_fast - ema_slow  # Differenz = MACD-Linie
    macd_signal = exponential_moving_average(macd, signal)  # Signallinie
    macd_hist = macd - macd_signal  # Histogramm als Differenz
    return macd, macd_signal, macd_hist  # drei Serien zurückgeben


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    """Bollinger-Bänder mit Mittenband und Bandbreite.

    Parameters
    ----------
    close : pd.Series
        Schlusskurse.
    window : int, optional
        Fenster für Mittelwert und Standardabweichung.
    n_std : float, optional
        Anzahl Standardabweichungen für die Bänder.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series, pd.Series]
        Mittleres Band, oberes Band, unteres Band, relative Breite.
    """
    mid = simple_moving_average(close, window)  # gleitender Mittelwert
    std = _safe_rolling(close, window).std()  # Standardabweichung im Fenster
    upper = mid + n_std * std  # oberes Band
    lower = mid - n_std * std  # unteres Band
    width = (upper - lower) / mid.replace(0, np.nan)  # Bandbreite relativ zum Mittelwert
    return mid, upper, lower, width  # vier Serien zurückgeben


def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """CCI: Abweichung vom gleitenden Durchschnitt in Einheiten MAD.

    Parameters
    ----------
    high, low, close : pd.Series
        Tageshochs, -tiefs und Schlusskurse.
    period : int, optional
        Fensterlänge des Indikators.

    Returns
    -------
    pd.Series
        CCI-Werte.
    """
    tp = (high + low + close) / 3.0  # Typical Price als Mittel der Extrema
    sma_tp = tp.rolling(window=period, min_periods=period).mean()  # gleitender Mittelwert
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=False  # mittlere absolute Abweichung
    )
    denom = 0.015 * mad.replace(0, np.nan)  # Skalierungskonstante 0.015
    cci_val = (tp - sma_tp) / denom  # Normierte Abweichung
    cci_val.name = f"cci_{period}"  # sprechender Name
    return cci_val  # Serie zurück

def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """ADX samt positiver/negativer Richtungsindizes berechnen.

    Parameters
    ----------
    high, low, close : pd.Series
        Hoch-, Tief- und Schlusskurse.
    period : int, optional
        Fensterlänge.

    Returns
    -------
    pd.DataFrame
        Enthält ADX sowie positive/negative Richtungsindizes.
    """

    # True Range
    prev_close = close.shift(1)  # Vortagesschluss zum TR-Vergleich
    tr = pd.concat([
        high - low,  # interne Spanne
        (high - prev_close).abs(),  # Abstand High zu prev_close
        (low - prev_close).abs()  # Abstand Low zu prev_close
    ], axis=1).max(axis=1)  # max der drei Komponenten

    # Directional Movements
    up_move = high.diff()  # Aufwärtsbewegung
    down_move = -low.diff()  # Abwärtsbewegung (negiert)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)  # positives DM
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)  # negatives DM
    plus_dm = pd.Series(plus_dm, index=high.index)  # zurück in Series-Form
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Wilder smoothing via EMA(alpha=1/period)
    alpha = 1.0 / period  # Glättungsfaktor
    tr_sm = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()  # geglätteter TR
    plus_dm_sm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()  # geglättetes +DM
    minus_dm_sm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()  # geglättetes -DM

    # DIs
    eps = 1e-12  # numerischer Schutz gegen Division durch 0
    plus_di = 100.0 * (plus_dm_sm / (tr_sm.replace(0, np.nan)))  # +DI in %
    minus_di = 100.0 * (minus_dm_sm / (tr_sm.replace(0, np.nan)))  # -DI in %

    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan))  # Differenzmaß
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()  # ADX-Glättung

    out = pd.DataFrame({  # Ergebnisse bündeln
        f"adx_{period}": adx_val,
        f"plus_di_{period}": plus_di,
        f"minus_di_{period}": minus_di,
    }, index=close.index)
    return out  # DataFrame zurückgeben

DEFAULTS = {  # typische Standardparameter für Indikatoren
    "sma": [20, 60],  # zwei Fenster für SMA
    "rsi": 14,  # Periode für RSI
    "macd": (12, 26, 9),  # Fast/Slow/Signal-EMA
    "boll": (20, 2.0),  # Fenster und Std-Anzahl für Bollinger
    "adv": 20,  # Fenster für average dollar volume
    "cci": 20  # Periode für CCI
}

