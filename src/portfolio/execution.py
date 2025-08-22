from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ------------------------- rechnet halben Spread aus -------------------------
def half_spread_price(p_ref: pd.Series, side: pd.Series, spread: pd.Series) -> pd.Series:
    """
    Half-Spread-Adjust:
      Buy (side>0):  p_exec = p_ref * (1 + 0.5*spread)
      Sell(side<0):  p_exec = p_ref * (1 - 0.5*spread)
    """
    side = side.fillna(0)
    spread = spread.fillna(0).clip(lower=0)
    adj = np.where(side >= 0, 1.0 + 0.5*spread, 1.0 - 0.5*spread)
    return p_ref * adj


# ------------------------- Rundet Stücke: Käufe floor, Verkäufe ceil (Standard: ganze Stücke) -------------------------
def round_shares(q: pd.Series, lot: int = 1) -> pd.Series:
    q = q.fillna(0.0)
    out = pd.Series(0.0, index=q.index)
    buy, sell = q > 0, q < 0
    out[buy]  = np.floor(q[buy]  / lot) * lot
    out[sell] = np.ceil( q[sell] / lot) * lot
    return out



def apply_execution(
    prices: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    order_col: str = "delta_shares",
    use_tplus1: bool = True,
    use_cs_spread: bool = True,
    fixed_spread_bps: Optional[float] = None,
    allow_short: bool = False,   # nur Flag; Kappen/Bounds kommt später im Constraints-Modul
    lot_size: int = 1,
) -> pd.DataFrame:
    """
    T+1-Execution ohne Lookahead.

    Erwartet MultiIndex (date, asset):
      prices: Spalten mind. ['exec_ref_tplus1'] und ggf. ['open', 'spread_cs'].
      orders: Spalte order_col = signierte Stücke q_t (Trade, ausgelöst an Tag t).

    Rückgabe-DF mit Spalten:
      ['q','p_ref','p_exec','notional_abs','spread_cost']
    """
    idx = orders.index
    q_raw = orders[order_col].reindex(idx).fillna(0.0)
    q = round_shares(q_raw, lot=lot_size)

    p_ref_col = "exec_ref_tplus1" if use_tplus1 else "open"
    p_ref = prices[p_ref_col].reindex(idx)

    if use_cs_spread:
        spread = prices.get("spread_cs", 0).reindex(idx).fillna(0.0).clip(lower=0.0)
    else:
        spread = pd.Series(0.0, index=idx)
        if fixed_spread_bps is not None:
            spread = pd.Series(float(fixed_spread_bps) / 1e4, index=idx)

    p_exec = half_spread_price(p_ref, q, spread)
    notional_abs = (q.abs() * p_exec).astype(float)
    spread_cost  = (q.abs() * p_ref * 0.5 * spread).astype(float)

    out = pd.DataFrame({
        "q": q,
        "p_ref": p_ref,
        "p_exec": p_exec,
        "notional_abs": notional_abs,
        "spread_cost": spread_cost,
    }).sort_index()
    return out
