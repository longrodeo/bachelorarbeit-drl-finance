from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def _as_series(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    # broadcast scalar/array auf den Index
    return pd.Series(x, index=index, dtype=float)

# ------------------------- rechnet halben Spread aus -------------------------
def half_spread_price(p_ref: pd.Series, side: pd.Series, spread: pd.Series) -> pd.Series:
    """
    Half-Spread-Adjust:
      Buy (side>0):  p_exec = p_ref * (1 + 0.5*spread)
      Sell(side<0):  p_exec = p_ref * (1 - 0.5*spread)
    """
    p_ref = _as_series(p_ref, p_ref.index)
    side = _as_series(side, p_ref.index).fillna(0.0)
    spread = _as_series(spread, p_ref.index).fillna(0.0)

    sign = np.where(side.values > 0, 1.0,
                    np.where(side.values < 0, -1.0, 0.0))
    return p_ref * (1.0 + 0.5 * spread * sign)

# ------------------------- Rundet Stücke: Käufe floor, Verkäufe ceil (Standard: ganze Stücke) -------------------------
def round_shares(q: pd.Series, lot: int = 1) -> pd.Series:
    q = q.fillna(0.0)
    out = pd.Series(0.0, index=q.index)
    buy, sell = q > 0, q < 0
    out[buy]  = np.floor(q[buy]  / lot) * lot
    out[sell] = np.ceil( q[sell] / lot) * lot
    return out


def plan_execution_series(
    q: pd.Series,                 # signierte Stücke (+Buy, -Sell), Index = assets
    p_ref: pd.Series,             # Referenzpreis je Asset (T+1 Open)
    spread: Optional[pd.Series] = None,  # CS-Spread je Asset (dezimal, z.B. 0.001)
    *,
    fixed_spread_bps: Optional[float] = None,  # alternativ fixer Spread in bp
    cash_assets: Optional[set] = None,         # z. B. {"CASH"} => Spread=0
) -> pd.DataFrame:
    """
    Einheitliche Ausführungslogik (pure, vektorisiert):
    - berechnet p_exec (Half-Spread), notional_abs, spread_cost.
    - keine I/O, keine Seiteneffekte -> deterministisch reproduzierbar.

    Rückgabe-Spalten je Asset:
      ["q", "p_ref", "p_exec", "notional_abs", "spread_cost"]
    """
    idx = q.index
    q = _as_series(q, idx).astype(float)
    p_ref = _as_series(p_ref, idx).astype(float)

    if spread is not None:
        spread = _as_series(spread, idx).astype(float)
    else:
        if fixed_spread_bps is None:
            spread = pd.Series(0.0, index=idx, dtype=float)
        else:
            spread = pd.Series(float(fixed_spread_bps) / 1e4, index=idx, dtype=float)

    # CASH-Spreads auf 0 setzen (falls angegeben)
    if cash_assets:
        mask_cash = pd.Index(idx).isin(cash_assets)
        if mask_cash.any():
            spread = spread.mask(mask_cash, 0.0)

    side = np.sign(q).astype(float)
    p_exec = half_spread_price(p_ref, side, spread)
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



# Einzelfunktion zum debuggen

def apply_execution(
    prices: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    order_col: str = "delta_shares",
    ref_col: str = "execution_price_t_plus_1_open",
    spread_col: str = "bid_ask_spread_corwin_schultz",
    use_tplus1: bool = True,                 # falls du mal 'open' statt T+1 nutzt
    use_cs_spread: bool = True,
    fixed_spread_bps: Optional[float] = None,
    lot_size: int = 1,
) -> pd.DataFrame:
    """
    T+1-Execution ohne Lookahead; MultiIndex (date, asset).
    Rückgabe-Spalten: ['q','p_ref','p_exec','notional_abs','spread_cost']
    """
    # 1) Orders runden
    q_rounded = round_shares(orders[order_col].reindex(orders.index).fillna(0.0), lot=lot_size)

    outs = []
    for date, q_d in q_rounded.groupby(level="date", sort=True):
        q_d = q_d.droplevel("date")
        px_d = prices.xs(date, level="date")

        # 2) Referenzpreis
        if use_tplus1:
            p_ref_d = px_d[ref_col].reindex(q_d.index).astype(float)
        else:
            p_ref_d = px_d["open"].reindex(q_d.index).astype(float)

        # 3) Spread-Quelle
        if use_cs_spread:
            spread_d = px_d[spread_col].reindex(q_d.index).astype(float)
            fixed_bps = None
        else:
            spread_d = None
            fixed_bps = fixed_spread_bps

        # 4) Einheitliche Execution-Logik
        out_d = plan_execution_series(
            q=q_d, p_ref=p_ref_d, spread=spread_d,
            fixed_spread_bps=fixed_bps,
            cash_assets={"CASH"},
        )
        out_d.index = pd.MultiIndex.from_product([[date], out_d.index], names=["date", "asset"])
        outs.append(out_d)

    out = pd.concat(outs).sort_index()
    return out

