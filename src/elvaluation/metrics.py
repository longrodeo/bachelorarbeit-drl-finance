# src/eval/metrics.py
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd


# --------------------------- kleine Hilfsfunktionen ---------------------------

def _to_series(x: Iterable[float]) -> pd.Series:
    s = pd.Series(x).astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _norm_cdf(z: float) -> float:
    # Phi(z)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _norm_ppf(p: float) -> float:
    # Acklam-Approximation für Phi^{-1}(p), 0<p<1 (keine SciPy-Abhängigkeit)
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
           3.754408661907416e+00 ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return -num / den
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den
    q = p - 0.5
    r = q*q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return num / den


# --------------------------- Rendite-basierte KPIs ---------------------------

def sharpe_ratio(
    returns: Iterable[float],
    rf: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Annualisierte Sharpe Ratio aus periodischen Renditen."""
    r = _to_series(returns) - rf
    if r.std(ddof=1) == 0 or len(r) < 2:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    return sr * math.sqrt(periods_per_year)

def information_ratio(
    returns: Iterable[float],
    benchmark: Iterable[float],
    periods_per_year: int = 252
) -> float:
    """Annualisierte Information Ratio (aktive Rendite / aktive Vol)."""
    r = _to_series(returns)
    b = _to_series(benchmark)
    n = min(len(r), len(b))
    if n < 2:
        return float("nan")
    ar = (r.iloc[:n] - b.iloc[:n])
    if ar.std(ddof=1) == 0:
        return float("nan")
    ir = ar.mean() / ar.std(ddof=1)
    return ir * math.sqrt(periods_per_year)

def probabilistic_sharpe_ratio(
    returns: Iterable[float],
    sr_threshold: float = 0.0,
) -> float:
    """
    PSR nach López de Prado:
    PSR = Phi( (SR - SR*) * sqrt(n-1) / sqrt(1 - s*SR + ((k-1)/4)*SR^2) )
    SR, s, k basieren auf NICHT annualisierten Überschussrenditen.
    """
    r = _to_series(returns)
    n = len(r)
    if n < 2:
        return float("nan")
    s = float(r.skew())                 # Schiefe
    k = float(r.kurt())                 # Excess Kurtosis (Normalverteilung -> 0)
    if r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    denom = math.sqrt(max(1e-16, 1.0 - s*sr + ((k - 1.0)/4.0)*(sr**2)))
    z = (sr - sr_threshold) * math.sqrt(max(1, n - 1)) / denom
    return _norm_cdf(z)

def deflated_sharpe_ratio(
    returns: Iterable[float],
    n_trials: int,
    var_sr: Optional[float] = None
) -> float:
    """
    DSR: PSR mit deflationiertem SR*-Schwellenwert.
    SR* ≈ z_N * sqrt(Var[SR]), wobei z_N ≈ E[max von N Standardnormalen].
    Var[SR] wird (falls nicht übergeben) aus PSR-Denominator rückgerechnet.
    """
    if n_trials < 1:
        return float("nan")
    r = _to_series(returns)
    n = len(r)
    if n < 2 or r.std(ddof=1) == 0:
        return float("nan")
    s = float(r.skew())
    k = float(r.kurt())
    sr = r.mean() / r.std(ddof=1)

    if var_sr is None:
        # aus PSR-Formel: denom^2 = 1 - s*SR + ((k-1)/4)*SR^2
        denom2 = max(1e-16, 1.0 - s*sr + ((k - 1.0)/4.0)*(sr**2))
        var_sr = denom2 / max(1, (n - 1))  # Schätzer für Var[SR]

    # erwarteter Maximalwert von N Standardnormalen (gute Approximation)
    p = (n_trials - 0.375) / (n_trials + 0.25)
    zN = _norm_ppf(min(max(p, 1e-12), 1 - 1e-12))
    sr_star = zN * math.sqrt(var_sr)

    # DSR = PSR gegen SR*:
    denom = math.sqrt(max(1e-16, 1.0 - s*sr + ((k - 1.0)/4.0)*(sr**2)))
    z = (sr - sr_star) * math.sqrt(max(1, n - 1)) / denom
    return _norm_cdf(z)


# --------------------------- Drawdown & Time-under-Water ---------------------------

def drawdown_series(returns: Iterable[float]) -> pd.Series:
    """Drawdown-Zeitreihe aus periodischen Renditen."""
    r = _to_series(returns)
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd

def dd_tuw_percentiles(
    returns: Iterable[float],
    percentiles: Sequence[int] = (95,)
) -> Dict[str, float]:
    """
    DD_p: p-Perzentil der Drawdown-Tiefe (negativ, z. B. -0.15).
    TuW_p: p-Perzentil der Unterwasser-Dauer (in Perioden).
    """
    dd = drawdown_series(returns)
    # Drawdown-Episoden identifizieren
    is_uwater = dd < 0
    depths = []
    durations = []
    dur = 0
    cur_min = 0.0
    for val in dd:
        if val < 0:
            dur += 1
            if val < cur_min:
                cur_min = val
        elif dur > 0:
            depths.append(cur_min)
            durations.append(dur)
            dur = 0
            cur_min = 0.0
    if dur > 0:
        depths.append(cur_min)
        durations.append(dur)

    out: Dict[str, float] = {}
    if depths:
        for p in percentiles:
            out[f"dd{p}"] = float(np.percentile(depths, p))
    else:
        for p in percentiles:
            out[f"dd{p}"] = float("nan")
    if durations:
        for p in percentiles:
            out[f"tuw{p}"] = float(np.percentile(durations, p))
    else:
        for p in percentiles:
            out[f"tuw{p}"] = float("nan")
    return out


# --------------------------- HHI-Konzentration ---------------------------

def hhi_concentration(returns: Iterable[float]) -> Tuple[float, float]:
    """
    HHI der positiven und negativen Beitragsgewichte aus periodischen Renditen.
    HHI = Sum_i w_i^2, w_i = |r_i| / Sum_j |r_j| (jeweils separat für +/-).
    """
    r = _to_series(returns)
    pos = r[r > 0].values
    neg = -r[r < 0].values  # Betrag
    def _hhi(x: np.ndarray) -> float:
        if x.size == 0:
            return float("nan")
        w = x / x.sum()
        return float((w**2).sum())
    return _hhi(pos), _hhi(neg)

def hhi_time_between_events(timestamps: Sequence[pd.Timestamp | int | float]) -> float:
    """
    HHI der Zeitabstände zwischen Ereignissen (z. B. Trades/Signals).
    Übergib eine sortierte Liste/Serie von Zeitpunkten (Indexwerte).
    """
    if timestamps is None or len(timestamps) < 2:
        return float("nan")
    x = pd.Index(timestamps)
    # Differenzen in "Schritte" oder (bei DatetimeIndex) in Tage
    if isinstance(x, pd.DatetimeIndex):
        gaps = np.diff(x.view(np.int64))  # ns
        gaps = np.abs(gaps.astype(np.float64))
    else:
        gaps = np.diff(np.asarray(x, dtype=float))
        gaps = np.abs(gaps)
    if np.all(gaps == 0):
        return float("nan")
    w = gaps / gaps.sum()
    return float((w**2).sum())


# --------------------------- Ausführungskosten / Turnover ---------------------------

def execution_cost_metrics(
    turnover: Iterable[float],
    fee_bps: float,
    slippage_bps: float,
    returns: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """
    Einfache Implementierung:
    - Kostenrate_t = Turnover_t * (fee_bps + slippage_bps) / 1e4
    - RoEC = Sum(Gross-Renditen) / Sum(Kostenraten)
    - Return-per-Turnover = Sum(Netto-Renditen) / Sum(Turnover)
    """
    to = _to_series(turnover).clip(lower=0.0)
    if len(to) == 0:
        return {
            "avg_turnover": float("nan"),
            "total_turnover": 0.0,
            "avg_cost_per_turnover_bps": fee_bps + slippage_bps,
            "mean_cost_rate": float("nan"),
            "sum_cost_rate": float("nan"),
            "return_per_turnover": float("nan"),
            "roec": float("nan"),
        }
    cost_rate = to * (fee_bps + slippage_bps) / 1e4
    out = {
        "avg_turnover": float(to.mean()),
        "total_turnover": float(to.sum()),
        "avg_cost_per_turnover_bps": float(fee_bps + slippage_bps),
        "mean_cost_rate": float(cost_rate.mean()),
        "sum_cost_rate": float(cost_rate.sum()),
        "return_per_turnover": float("nan"),
        "roec": float("nan"),
    }
    if returns is not None:
        r = _to_series(returns)
        n = min(len(r), len(cost_rate))
        if n > 0:
            gross = float(r.iloc[:n].sum())
            net = float((r.iloc[:n] - cost_rate.iloc[:n]).sum())
            tot_to = float(to.iloc[:n].sum())
            out["return_per_turnover"] = net / (tot_to + 1e-12)
            out["roec"] = gross / (float(cost_rate.iloc[:n].sum()) + 1e-12)
    return out


# --------------------------- Bequemer Sammel-Output ---------------------------

def summarize_core_kpis(
    returns: Iterable[float],
    benchmark: Optional[Iterable[float]] = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
    psr_threshold: float = 0.0,
    n_trials_for_dsr: Optional[int] = None,
    turnover: Optional[Iterable[float]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Dict[str, float]:
    """Kompakte Auswertung häufig genutzter Kennzahlen (für HPO/Logging)."""
    r = _to_series(returns) - rf
    out: Dict[str, float] = {
        "sr": sharpe_ratio(r, 0.0, periods_per_year),
        "psr": probabilistic_sharpe_ratio(r, psr_threshold),
    }
    if n_trials_for_dsr is not None and n_trials_for_dsr > 0:
        out["dsr"] = deflated_sharpe_ratio(r, n_trials_for_dsr)
    if benchmark is not None:
        out["ir"] = information_ratio(r, _to_series(benchmark) - rf, periods_per_year)
    out.update(dd_tuw_percentiles(r, percentiles=(95,)))
    hhi_pos, hhi_neg = hhi_concentration(r)
    out["hhi_pos"] = hhi_pos
    out["hhi_neg"] = hhi_neg
    if turnover is not None:
        out.update(execution_cost_metrics(turnover, fee_bps, slippage_bps, returns=r))
    return out
