from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _s(x: Iterable[float]) -> pd.Series:
    s = pd.Series(x, dtype=float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def simple_from_log(log_returns: Iterable[float]) -> pd.Series:
    """Convert log returns to simple returns: r = exp(r_log) - 1."""
    rlog = _s(log_returns)
    return pd.Series(np.expm1(rlog.to_numpy()), index=rlog.index, dtype=float)


def cum_return(returns: Iterable[float]) -> float:
    r = _s(returns)
    if len(r) == 0:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def ann_return(returns: Iterable[float], periods_per_year: int = 252) -> float:
    r = _s(returns)
    n = len(r)
    if n == 0:
        return float("nan")
    cr = (1.0 + r).prod()
    return float(cr ** (periods_per_year / n) - 1.0)


def sharpe_hat(returns: Iterable[float]) -> float:
    r = _s(returns)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(r.mean() / sd)


def sharpe_ann(returns: Iterable[float], periods_per_year: int = 252) -> float:
    sr = sharpe_hat(returns)
    if not np.isfinite(sr):
        return float("nan")
    return float(sr * math.sqrt(periods_per_year))


def vol_ann(returns: Iterable[float], periods_per_year: int = 252) -> float:
    r = _s(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sortino_ann(returns: Iterable[float], periods_per_year: int = 252, mar: float = 0.0) -> float:
    r = _s(returns)
    if len(r) < 2:
        return float("nan")
    downside = np.minimum(0.0, (r - mar).to_numpy())
    dd = float(np.sqrt(np.mean(downside * downside)))
    if dd == 0.0:
        return float("nan")
    return float(((r.mean() - mar) / dd) * math.sqrt(periods_per_year))


def cvar(returns: Iterable[float], alpha: float = 0.95) -> float:
    r = _s(returns)
    if len(r) == 0:
        return float("nan")
    q = float(r.quantile(1.0 - alpha))
    tail = r[r <= q]
    if len(tail) == 0:
        return float("nan")
    return float(tail.mean())


def maxdd(returns: Iterable[float]) -> float:
    r = _s(returns)
    if len(r) == 0:
        return float("nan")
    w = (1.0 + r).cumprod()
    peak = w.cummax()
    dd = w / peak - 1.0
    return float(dd.min())


def calmar(returns: Iterable[float], periods_per_year: int = 252) -> float:
    ar = ann_return(returns, periods_per_year=periods_per_year)
    mdd = maxdd(returns)
    if not np.isfinite(ar) or not np.isfinite(mdd) or mdd == 0.0:
        return float("nan")
    return float(ar / abs(mdd))


# ---- Lopez: PSR/DSR (ohne SciPy) ----
def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ppf(p: float) -> float:
    # Acklam approximation (inverse standard normal CDF)
    if not (0.0 < p < 1.0):
        raise ValueError("p in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def psr(excess_returns: Iterable[float], sr_threshold: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Lopez): excess returns, nicht annualisiert.
    Achtung: pandas.kurt() ist i.d.R. excess kurtosis -> +3 für Normal=3.
    """
    r = _s(excess_returns)
    t = len(r)
    if t < 2:
        return float("nan")

    sr = sharpe_hat(r)
    if not np.isfinite(sr):
        return float("nan")

    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0

    denom_sq = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * (sr ** 2)
    denom = math.sqrt(max(1e-16, denom_sq))

    z = (sr - sr_threshold) * math.sqrt(t - 1.0) / denom
    return float(_phi(z))


def dsr(excess_returns: Iterable[float], n_trials: int, sr_var_trials: Optional[float]) -> float:
    """
    Deflated Sharpe Ratio (Lopez): benötigt
      - n_trials (Anzahl getesteter Varianten)
      - sr_var_trials (Varianz der NICHT annualisierten Sharpe-Schätzer über Trials)
    """
    if n_trials is None or n_trials < 1:
        return float("nan")
    if sr_var_trials is None or not np.isfinite(sr_var_trials) or sr_var_trials < 0.0:
        return float("nan")

    gamma = 0.5772156649015329
    N = float(n_trials)

    z1 = _ppf(1.0 - 1.0 / N)
    z2 = _ppf(1.0 - 1.0 / (N * math.e))
    sr_star = math.sqrt(sr_var_trials) * ((1.0 - gamma) * z1 + gamma * z2)

    return float(psr(excess_returns, sr_threshold=sr_star))


def scorecard_eval(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    alpha_cvar: float = 0.95,
) -> Dict[str, float]:
    r = _s(returns)
    return {
        "cum_return": cum_return(r),
        "ann_return": ann_return(r, periods_per_year),
        "sharpe": sharpe_ann(r, periods_per_year),
        "sortino": sortino_ann(r, periods_per_year, mar=0.0),
        "vol_ann": vol_ann(r, periods_per_year),
        "maxdd": maxdd(r),
        "calmar": calmar(r, periods_per_year),
        "cvar": cvar(r, alpha=alpha_cvar),
    }
