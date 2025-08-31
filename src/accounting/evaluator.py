# src/accounting/evaluator.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from accounting.reward import RewardSpec
from utils.parquet_io import load_parquet, save_parquet

EPS = 1e-12


# --------- Hilfsfunktionen: (C)VaR & CVaR-Serien --------------------------------

def _var_cvar_from_array(x: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Empirischer VaR/CVaR der linken Tail (alpha in (0,1)).
    x: 1D-Array der (Log-)Returns.
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    var = float(np.quantile(x, alpha, method="linear"))
    tail = x[x <= var]
    if tail.size == 0:
        cvar = var
    else:
        cvar = float(np.mean(tail))
    return var, cvar


def _rolling_cvar(
    r: pd.Series,
    *,
    alpha: float,
    window: int,
    include_current: bool,  # ex_post=True, ex_ante=False
    ewm_alpha: float | None = None,
) -> pd.Series:
    """
    Rolling-CVaR-Serie. Für ex-ante wird die Return-Serie vor dem Rolling um 1 nach unten geshiftet,
    so dass das Fenster bis t-1 endet. Für ex-post wird nicht geshiftet (Fenster endet bei t).
    Optional: ewm-Glättung auf der CVaR-Serie.
    """
    r_use = r if include_current else r.shift(1)
    def _cvar_window(a: np.ndarray) -> float:
        _, cvar = _var_cvar_from_array(a, alpha)
        return cvar

    cvar = r_use.rolling(window=window, min_periods=window).apply(
        lambda w: _cvar_window(w.astype(float)), raw=True
    )

    if ewm_alpha is not None:
        cvar = cvar.ewm(alpha=float(ewm_alpha), adjust=False).mean()
    return cvar


# --------- MDD / ΔMDD ------------------------------------------------------------

def _mdd_series(nav: pd.Series) -> pd.Series:
    """
    Maximum Drawdown per Zeitpunkt auf Basis des NAV-Pfads.
    MDD_t = (Peak_t - NAV_t)/Peak_t, Peak_t = cummax(NAV).
    """
    peak = nav.cummax()
    mdd = (peak - nav) / peak.clip(lower=EPS)
    return mdd


# --------- Hauptfunktion: Rewards aus Snapshots ----------------------------------

def compute_rewards_from_snapshots(
    accounting_dir: Path,
    *,
    spec: RewardSpec = RewardSpec("log"),
    out_name: str = "rewards.parquet",
) -> pd.DataFrame:
    """
    Liest portfolio_snapshots.parquet (aus dem AccountingRecorder) und erzeugt
    eine Tabelle mit NAV_t, NAV_{t+1}, r_log, (I)CVaR, ΔMDD und Reward.
    Speichert unter accounting_dir/out_name und gibt das DataFrame zurück.

    Erwartet im snapshots-Parquet mindestens:
      - round (int), t (Timestamp), t_plus_1 (Timestamp)
      - portfolio_value_t1 (float)  -> NAV_{t+1} pro Round (einmalig; falls je Asset dupliziert: wir deduplizieren)
    """
    accounting_dir = Path(accounting_dir)
    if not accounting_dir.is_absolute():
        root = Path(__file__).resolve().parents[2]
        accounting_dir = root / "data" / accounting_dir

    snaps_path = accounting_dir / "portfolio_snapshots.parquet"
    if not snaps_path.exists():
        raise FileNotFoundError(f"{snaps_path} nicht gefunden.")

    # 1) Snapshots laden & auf Round-Ebene reduzieren
    snaps = load_parquet(snaps_path)
    # Eine Zeile je Round (falls im Snapshot je Asset dupliziert ist)
    base = (
        snaps[["round", "t", "t_plus_1", "portfolio_value_t1"]]
        .drop_duplicates(subset=["round"])
        .sort_values("round")
        .reset_index(drop=True)
    )
    base = base.rename(columns={"portfolio_value_t1": "nav_t1"})
    # NAV_t = Shift von NAV_{t+1}
    base["nav_t"] = base["nav_t1"].shift(1)

    # 2) r_log_t (additiv)
    base["r_log_t"] = np.log(base["nav_t1"].clip(lower=EPS) / base["nav_t"].clip(lower=EPS))

    # 3) CVaR/ICVaR je nach Spec
    if spec.kind in ("icvar", "icvar_dd"):
        include_current = (spec.icvar_mode == "ex_post")
        cvar = _rolling_cvar(
            base["r_log_t"],
            alpha=spec.alpha,
            window=spec.window,
            include_current=include_current,
            ewm_alpha=spec.ewm_alpha,
        )
        base["cvar_t"] = cvar
        base["cvar_tminus1"] = base["cvar_t"].shift(1)
        base["icvar_t"] = base["cvar_t"] - base["cvar_tminus1"]
        # Warm-up: fehlende Werte (NaN) setzen wir konservativ auf 0 (keine Strafe am Anfang)
        base["icvar_t"] = base["icvar_t"].fillna(0.0)
    else:
        base["icvar_t"] = 0.0

    # 4) ΔMDD_t (nur für icvar_dd relevant, sonst 0)
    if spec.kind == "icvar_dd":
        # MDD auf NAV_t- und NAV_{t+1}-Pfad (beide aus derselben Round-Tabelle baubar)
        mdd_t = _mdd_series(base["nav_t"].ffill())
        mdd_t1 = _mdd_series(base["nav_t1"].ffill())
        # bis t+1
        delta = (mdd_t1 - mdd_t).clip(lower=0.0)                           # nur Verschlechterungen
        base["delta_mdd_t"] = delta.fillna(0.0)
    else:
        base["delta_mdd_t"] = 0.0

    # 5) Reward je Spec
    if spec.kind == "log":
        base["reward_t"] = base["r_log_t"]
    elif spec.kind == "icvar":
        base["reward_t"] = base["r_log_t"] - spec.lambda_ * base["icvar_t"]
    elif spec.kind == "icvar_dd":
        base["reward_t"] = base["r_log_t"] - spec.lambda_ * base["icvar_t"] - spec.gamma * base["delta_mdd_t"]
    else:
        raise ValueError(f"Unbekannte Reward-Variante: {spec.kind}")

    # 6) Clean-up: erste Round hat nav_t = NaN -> r_log_t/Reward nicht definiert => droppen
    out = base.dropna(subset=["nav_t"]).copy()

    # 7) Metadaten/Parameter anhängen (nützlich für spätere Auswertungen)
    out["reward_kind"] = spec.kind
    out["icvar_mode"] = spec.icvar_mode
    out["alpha"] = spec.alpha
    out["window"] = spec.window
    out["lambda_"] = spec.lambda_
    out["gamma"] = spec.gamma
    out["estimator"] = spec.estimator
    out["ewm_alpha"] = spec.ewm_alpha

    # 8) Persistieren
    save_parquet(out, accounting_dir / out_name)
    return out
