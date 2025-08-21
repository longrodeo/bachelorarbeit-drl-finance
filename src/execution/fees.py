from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict

def load_costs(costs_path: Path | str) -> Dict:
    return yaml.safe_load(Path(costs_path).read_text(encoding="utf-8"))

def _bps_to_frac(x: float) -> float:
    return float(x) / 1e4

def apply_fees(
    trades: pd.DataFrame,
    *,
    commission_bps: float = 0.0,
    min_fee_abs: float = 0.0,
    use_vol_slippage: bool = False,
    sigma_hl: Optional[pd.Series] = None,
    k_bps_per_sigma: float = 0.0,
) -> pd.DataFrame:
    """
    Fügt 'fees' (Kommission) und optional 'vol_slip' hinzu.
    Erwartet trades mit Spalten: ['q','p_exec','notional_abs','spread_cost'].

    - Kommission: fees = notional_abs * commission_bps/1e4
    - Mindestgebühr: je (date, asset) max(fees, min_fee_abs)  [optional]
    - Optionale Vol-Slippage: vol_slip = |q|*p_ref * (k_bps_per_sigma*sigma_hl)/1e4
      (nur wenn use_vol_slippage=True und sigma_hl übergeben)
    """
    out = trades.copy()
    fees = out["notional_abs"] * _bps_to_frac(commission_bps)
    if min_fee_abs and min_fee_abs > 0:
        fees = fees.clip(lower=float(min_fee_abs))

    out["fees"] = fees

    if use_vol_slippage and sigma_hl is not None:
        # p_ref kann aus trades stammen, sonst 0
        p_ref = out.get("p_ref", 0)
        vol_bps = float(k_bps_per_sigma) * sigma_hl.reindex(out.index).fillna(0.0)
        out["vol_slip"] = (out["q"].abs() * p_ref * _bps_to_frac(vol_bps)).astype(float)
    else:
        out["vol_slip"] = 0.0

    out["total_cost"] = out["spread_cost"] + out["fees"] + out["vol_slip"]
    return out

def apply_fees_from_config(trades: pd.DataFrame, costs_path: Path | str) -> pd.DataFrame:
    """Liest Kostenparameter aus `costs_path` und wendet sie auf ``trades`` an."""
    params = load_costs(costs_path)
    return apply_fees(trades, **params)
