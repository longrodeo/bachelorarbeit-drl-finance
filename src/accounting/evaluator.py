# src/accounting/evaluator.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from accounting.reward import RewardSpec
from utils.parquet_io import load_parquet, save_parquet

EPS = 1e-12


# --------- Hilfsfunktionen: (C)VaR & CVaR-Serien --------------------------------

def mts_var_cvar_icvar(
    r: pd.Series,
    *,
    alpha: float = 0.05,        # Tail-Masse (z.B. 0.05 = 5%-Left-Tail)
    min_period: int = 1,        # ab wie vielen Punkten schätzen
    include_current: bool = True,  # ex_post=True, ex_ante=False
    ewm_alpha: float | None = None, # optional Glättung auf CVaR
    as_series: bool = False
):
    """
    MTS-Definition auf VERLUST-Skala (>=0):
      VaR_{α,t} = - Quantil_α({X_1..X_t})
      CVaR_{α,t} = VaR_{α,t} + (1/(α t)) * sum_{k<=t} max( (-X_k) - VaR_{α,k}, 0 )
      ICVaR_{α,t} = CVaR_{α,t} - CVaR_{α,t-1}

    r : Return-Serie X_t (negatives Tail ist 'bad'), zeitlich sortiert.
    Gibt zurück: (VaR_t_loss>=0, CVaR_t_loss>=0, ICVaR_t_loss)
    """
    r_use = r if include_current else r.shift(1)             # ex-post vs ex-ante
    # Anzahl gültiger Beobachtungen (t)
    n = r_use.expanding().count()

    # (1) VaR_k (auf LOSS-Skala) via Return-Quantil
    var_ret = r_use.expanding(min_periods=min_period).apply(
        lambda a: np.quantile(a[np.isfinite(a)], alpha, method="linear"),
        raw=True
    )
    var_loss = -var_ret  # positiv

    # (2) Exzessverluste relativ zu VaR_k
    L = -r_use
    excess = (L - var_loss).clip(lower=0)

    # (3) CVaR_t: VaR_t + (kumul. Exzess)/(α * t)
    cum_excess = excess.fillna(0).cumsum()
    cvar_loss = var_loss + (cum_excess / (alpha * n)).where(n >= min_period)

    # optional Glättung nur auf der CVaR-Serie
    if ewm_alpha is not None:
        cvar_loss = cvar_loss.ewm(alpha=float(ewm_alpha), adjust=False).mean()

    icvar = cvar_loss.diff()

    if as_series:
        return var_loss, cvar_loss, icvar

    # --- Skalare (nur t & t-1) ---
    s = cvar_loss.dropna()
    if len(s) == 0:
        return 0.0, 0.0, 0.0
    if len(s) == 1:
        x = float(s.iloc[-1])
        return 0.0, x, x
    cvar_t = float(s.iloc[-1])
    cvar_tm1 = float(s.iloc[-2])
    return cvar_t - cvar_tm1, cvar_t, cvar_tm1


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
        snaps[["round", "t", "portfolio_value_t"]]
        .drop_duplicates(subset=["round"])
        .sort_values("round")
        .reset_index(drop=True)
    )
    base = base.rename(columns={"portfolio_value_t": "nav_t"})
    # NAV_t = Shift von NAV_{t+1}
    base["nav_t-1"] = base["nav_t"].shift(1)

    # 2) r_log_t (additiv)
    base["r_log_t"] = np.log(base["nav_t"].clip(lower=EPS) / base["nav_t-1"].clip(lower=EPS))

    # 3) CVaR/ICVaR je nach Spec
    if spec.kind in ("icvar", "icvar_dd"):
        include_current = (spec.icvar_mode == "ex_post")
        var ,cvar, icvar = mts_var_cvar_icvar(
            base["r_log_t"],
            alpha=spec.alpha,
            min_period=spec.min_period,
            include_current=include_current,
            ewm_alpha=spec.ewm_alpha,
            as_series=True
        )
        base["var_t"] = var
        base["cvar_t"] = cvar
        base["cvar_tminus1"] = base["cvar_t"].shift(1)
        base["icvar_t"] = icvar
        # Warm-up: fehlende Werte (NaN) setzen wir konservativ auf 0 (keine Strafe am Anfang)
        base["icvar_t"] = base["icvar_t"].fillna(0.0)
    else:
        base["icvar_t"] = 0.0

    # 4) ΔMDD_t (nur für icvar_dd relevant, sonst 0)
    if spec.kind == "icvar_dd":
        # MDD auf NAV_t- und NAV_{t+1}-Pfad (beide aus derselben Round-Tabelle baubar)
        mdd_t = _mdd_series(base["nav_t-1"].ffill())
        mdd_t1 = _mdd_series(base["nav_t"].ffill())
        # bis t
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
    out = base.dropna(subset=["nav_t-1"]).copy()

    # 7) Metadaten/Parameter anhängen (nützlich für spätere Auswertungen)
    out["reward_kind"] = spec.kind
    out["icvar_mode"] = spec.icvar_mode
    out["alpha"] = spec.alpha
    out["lambda_"] = spec.lambda_
    out["gamma"] = spec.gamma
    out["estimator"] = spec.estimator
    out["ewm_alpha"] = spec.ewm_alpha

    # 8) Persistieren
    save_parquet(out, accounting_dir / out_name)
    return out
