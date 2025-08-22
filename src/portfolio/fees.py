from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional

def load_costs(costs_path: Path | str) -> Dict:
    return yaml.safe_load(Path(costs_path).read_text(encoding="utf-8"))

def _bps(x: float) -> float:
    return float(x) / 1e4

def apply_fees(
    trades: pd.DataFrame,
    *,
    commission_bps: float = 0.0,
    use_vol_slippage: bool = False,
    sigma_hl: Optional[pd.Series] = None,
    k_bps_per_sigma: float = 0.0,
) -> pd.DataFrame:
    """
    Gebühren (bps auf Notional) + optional Vol-Slippage.
    Erwartet trades mit Spalten: ['q','p_ref','p_exec','notional_abs','spread_cost'].
    """
    out = trades.copy()

    # Kommission
    fees = out["notional_abs"] * _bps(commission_bps)
    out["fees"] = fees

    # optionale Vol-Slippage proportional zu sigma_hl
    if use_vol_slippage and sigma_hl is not None:
        vol_bps = float(k_bps_per_sigma) * sigma_hl.reindex(out.index).fillna(0.0)
        out["vol_slip"] = (out["q"].abs() * out.get("p_ref", 0) * _bps(vol_bps)).astype(float)
    else:
        out["vol_slip"] = 0.0

    out["total_cost"] = out["spread_cost"] + out["fees"] + out["vol_slip"]
    return out
