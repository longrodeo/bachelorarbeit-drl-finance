# src/eval/kpis.py
from __future__ import annotations
import math
from typing import Iterable, Optional, Dict
import numpy as np
import pandas as pd

# ----------------- Helpers -----------------
def _s(x: Iterable[float]) -> pd.Series:
    s = pd.Series(x, dtype=float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _ppf(p: float) -> float:
    # Acklam-Approximation für Phi^{-1}(p) ohne SciPy
    if not (0.0 < p < 1.0): raise ValueError("p in (0,1)")
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1-0.02425
    if p < pl:
        q=math.sqrt(-2*math.log(p)); num=(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]); den=((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return -num/den
    if p > ph:
        q=math.sqrt(-2*math.log(1-p)); num=(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]); den=((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        return num/den
    q=p-0.5; r=q*q
    num=(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
    den=(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return num/den

# ----------------- Kern-KPIs -----------------
def sharpe_ann(returns: Iterable[float], periods_per_year: int = 252) -> float:
    r = _s(returns)
    if len(r) < 2: return float("nan")
    sd = r.std(ddof=1)
    if sd == 0: return float("nan")
    return float((r.mean() / sd) * math.sqrt(periods_per_year))

def vol_ann(returns: Iterable[float], periods_per_year: int = 252) -> float:
    r = _s(returns)
    if len(r) < 2: return float("nan")
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))

def psr(returns: Iterable[float], sr_threshold: float = 0.0) -> float:
    r = _s(returns)
    n = len(r)
    if n < 2: return float("nan")
    sd = r.std(ddof=1)
    if sd == 0: return float("nan")
    s, k = float(r.skew()), float(r.kurt())
    sr = float(r.mean() / sd)
    denom = math.sqrt(max(1e-16, 1.0 - s*sr + ((k - 1.0)/4.0)*(sr**2)))
    z = (sr - sr_threshold) * math.sqrt(max(1, n - 1)) / denom
    return float(_phi(z))

def dsr(returns: Iterable[float], n_trials: int) -> float:
    if n_trials < 1: return float("nan")
    r = _s(returns)
    n = len(r)
    if n < 2: return float("nan")
    sd = r.std(ddof=1)
    if sd == 0: return float("nan")
    s, k = float(r.skew()), float(r.kurt())
    sr = float(r.mean() / sd)
    denom2 = max(1e-16, 1.0 - s*sr + ((k - 1.0)/4.0)*(sr**2))
    var_sr = denom2 / max(1, (n - 1))
    p = (n_trials - 0.375) / (n_trials + 0.25)
    zN = _ppf(min(max(p, 1e-12), 1 - 1e-12))
    sr_star = zN * math.sqrt(var_sr)
    z = (sr - sr_star) * math.sqrt(max(1, n - 1)) / math.sqrt(denom2)
    return float(_phi(z))

def _dd_series(returns: Iterable[float]) -> pd.Series:
    r = _s(returns)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    return eq / peak - 1.0

def maxdd(returns: Iterable[float]) -> float:
    dd = _dd_series(returns)
    return float(dd.min()) if len(dd) else float("nan")

def dd_tuw_95(returns: Iterable[float]) -> Dict[str, float]:
    dd = _dd_series(returns)
    depths, durs = [], []
    cur_min, dur = 0.0, 0
    for v in dd:
        if v < 0:
            dur += 1
            cur_min = min(cur_min, float(v))
        elif dur > 0:
            depths.append(cur_min); durs.append(dur)
            cur_min, dur = 0.0, 0
    if dur > 0: depths.append(cur_min); durs.append(dur)
    return {
        "dd95": float(np.percentile(depths, 95)) if depths else float("nan"),
        "tuw95": float(np.percentile(durs,   95)) if durs   else float("nan"),
    }

def total_cost(
    cost_rate: Optional[Iterable[float]] = None,
    *,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> float:
    """
    Summe expliziter Kosten. Entweder cost_rate übergeben (bevorzugt),
    oder aus Turnover und bps (Gebühr+Slippage) berechnen.
    """
    if cost_rate is not None:
        c = _s(cost_rate)
        return float(c.sum())
    if turnover is None:
        return 0.0
    to = _s(turnover).clip(lower=0.0)
    return float((to * (fee_bps + slippage_bps) / 1e4).sum())

def roec(
    returns: Iterable[float],
    cost_rate: Optional[Iterable[float]] = None,
    *,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> float:
    """
    Return on Execution Costs = Sum(Gross-Returns) / Sum(Kosten).
    Kosten via cost_rate ODER via Turnover+bps.
    """
    r = _s(returns)
    if len(r) == 0: return float("nan")
    if cost_rate is not None:
        c = _s(cost_rate)
        n = min(len(r), len(c))
        cost_sum = float(c.iloc[:n].sum())
        gross_sum = float(r.iloc[:n].sum())
        return float(gross_sum / (cost_sum + 1e-12))
    if turnover is None:
        return float("nan")
    to = _s(turnover).clip(lower=0.0)
    n = min(len(r), len(to))
    cost_sum = float((to.iloc[:n] * (fee_bps + slippage_bps) / 1e4).sum())
    gross_sum = float(r.iloc[:n].sum())
    return float(gross_sum / (cost_sum + 1e-12))

def cum_return_net(
    returns: Iterable[float],
    cost_rate: Optional[Iterable[float]] = None,
    *,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> float:
    """
    Nettokumulierte Rendite: Produkt(1 + r_t - cost_t) - 1.
    cost_t aus cost_rate oder aus Turnover+bps.
    """
    r = _s(returns)
    if cost_rate is not None:
        c = _s(cost_rate).reindex(r.index).fillna(0.0)
        rr = r - c
        return float((1.0 + rr).prod() - 1.0)
    if turnover is None:
        return float((1.0 + r).prod() - 1.0)
    to = _s(turnover).clip(lower=0.0).reindex(r.index).fillna(0.0)
    c = to * (fee_bps + slippage_bps) / 1e4
    rr = r - c
    return float((1.0 + rr).prod() - 1.0)

# ----------------- Kompakte Scorecards -----------------
def scorecard_train(
    returns: Iterable[float],
    *,
    n_trials_for_dsr: int,
    cost_rate: Optional[Iterable[float]] = None,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> Dict[str, float]:
    """
    HPO/Training-Set: DSR + (DD95, TuW95) + RoEC + CumReturn_net.
    Returns sollten Überschussrenditen sein (ggf. vs. risikofrei).
    """
    out = {"dsr": dsr(returns, n_trials_for_dsr)}
    out.update(dd_tuw_95(returns))
    out["roec"] = roec(returns, cost_rate, turnover=turnover, fee_bps=fee_bps, slippage_bps=slippage_bps)
    out["cum_return"] = cum_return_net(returns, cost_rate, turnover=turnover, fee_bps=fee_bps, slippage_bps=slippage_bps)
    return out

def scorecard_baseline(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    cost_rate: Optional[Iterable[float]] = None,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0
) -> Dict[str, float]:
    """
    Kompakter Report für Baseline/Reader: CumReturn_net, Sharpe, Vol, MaxDD, TotalCost, RoEC.
    """
    out = {
        "cum_return": cum_return_net(returns, cost_rate, turnover=turnover, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "sharpe": sharpe_ann(returns, periods_per_year),
        "vol_ann": vol_ann(returns, periods_per_year),
        "maxdd": maxdd(returns),
        "total_cost": total_cost(cost_rate, turnover=turnover, fee_bps=fee_bps, slippage_bps=slippage_bps),
        "roec": roec(returns, cost_rate, turnover=turnover, fee_bps=fee_bps, slippage_bps=slippage_bps),
    }
    return out
