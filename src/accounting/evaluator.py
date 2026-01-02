# src/accounting/evaluator.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.accounting.reward import RewardSpec, apply_reward_spec
from src.utils.parquet_io import load_parquet, save_parquet



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
    base["reward_t"] = apply_reward_spec(
        r_log_t=base["r_log_t"],
        icvar_t=base["icvar_t"],
        delta_mdd_t=base["delta_mdd_t"],
        spec=spec,
    )

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

# --- OnlineEvaluator: läuft ohne Recorder/Parquet ----------------------------

class OnlineEvaluator:
    """
    Minimaler Online-Evaluator für Test-Episoden ohne Recorder.
    Füttern mit NAV_t und r_log_t (falls r_log nicht gegeben, aus NAV ableiten).
    Am Ende: gleiche Größen wie compute_rewards_from_snapshots, aber im RAM.
    """
    def __init__(self, *, kind: str = "log",
                 alpha: float = 0.05, min_period: int = 1,
                 icvar_mode: str = "ex_post", ewm_alpha: float | None = None,
                 lambda_: float = 1.0, gamma: float = 0.0):
        self.kind = str(kind)
        self.alpha = float(alpha)
        self.min_period = int(min_period)
        self.icvar_mode = str(icvar_mode)
        self.ewm_alpha = ewm_alpha if ewm_alpha is None else float(ewm_alpha)
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self._nav = []   # NAV_t
        self._r = []     # r_log_t

    def update(self, *, nav_t: float | None = None, r_log_t: float | None = None):
        if (nav_t is None) and (r_log_t is None):
            return
        if nav_t is not None:
            self._nav.append(float(nav_t))
            # Wenn r nicht explizit kommt, aus NAV ableiten (additiv):
            if r_log_t is None and len(self._nav) >= 2:
                a, b = self._nav[-2], self._nav[-1]
                r_log_t = float(np.log(max(b, EPS) / max(a, EPS)))
        if r_log_t is not None:
            self._r.append(float(r_log_t))

    def finalize(self) -> pd.DataFrame:
        nav = pd.Series(self._nav, name="nav_t").reset_index(drop=True)
        # r-Serie: n-1 Einträge (ab 2. NAV), deckt sich mit Snapshot-Logik
        if len(self._r) == len(nav):
            r = pd.Series(self._r)
        else:
            r = pd.Series(self._r[:max(0, len(nav)-1)])
        base = pd.DataFrame({
            "nav_t": nav,
            "nav_t-1": nav.shift(1),
        })
        base["r_log_t"] = r.reindex(base.index).astype(float)

        # (I)CVaR nach Spec
        if self.kind in ("icvar", "icvar_dd"):
            include_current = (self.icvar_mode.lower() == "ex_post")
            var_s, cvar_s, icvar_s = mts_var_cvar_icvar(
                base["r_log_t"],
                alpha=self.alpha,
                min_period=self.min_period,
                include_current=include_current,
                ewm_alpha=self.ewm_alpha,
                as_series=True
            )
            base["var_t"] = var_s
            base["cvar_t"] = cvar_s
            base["cvar_tminus1"] = base["cvar_t"].shift(1)
            base["icvar_t"] = icvar_s.fillna(0.0)
        else:
            base["icvar_t"] = 0.0

        # ΔMDD (nur icvar_dd)
        if self.kind == "icvar_dd":
            mdd_t  = _mdd_series(base["nav_t-1"].ffill())
            mdd_t1 = _mdd_series(base["nav_t"].ffill())
            base["delta_mdd_t"] = (mdd_t1 - mdd_t).clip(lower=0.0).fillna(0.0)
        else:
            base["delta_mdd_t"] = 0.0

        # Reward
        spec = RewardSpec(kind=self.kind, lambda_=self.lambda_, gamma=self.gamma)
        base["reward_t"] = apply_reward_spec(
            r_log_t=base["r_log_t"],
            icvar_t=base["icvar_t"],
            delta_mdd_t=base["delta_mdd_t"],
            spec=spec,
        )

        out = base.dropna(subset=["nav_t-1"]).reset_index(drop=True)
        return out

