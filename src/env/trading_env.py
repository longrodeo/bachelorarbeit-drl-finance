# -*- coding: utf-8 -*-
"""
TradingEnv – Gymnasium-Env (Dict-Obs)

- State:   state_builder.build_state_for_date(...)
- Exec:    portfolio.step(px_t1, w_target, cash_factor)   # Turnover/Fees/Spread/Cash IM BROKER
- Preise:  Exec auf adj_open[t+1], Bewertung via adj_close[t+1] (beides brokerseitig)
- Reward:  "log" | "icvar" | "icvar_dd" (ICVaR + optional ΔMDD), danach optional reward_scaler.update(...)
- Obs:     Builder-Outputs + "position" (A+1 inkl. Cash), damit Agent Cash explizit sieht
- Done:    truncated am Datenende, terminated bei value <= 0
"""

from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        # --- Daten & State-Builder ---
        panel_clean: pd.DataFrame,      # MultiIndex (date, asset), enthält mind. adj_open/adj_close/spread...
        dates: pd.DatetimeIndex,        # alle Handelstage (== level "date" im panel)
        assets: list[str],              # feste Asset-Reihenfolge
        spec: Any,                      # StateSpec für build_state_for_date
        state_builder: Any,             # Objekt/Modul mit build_state_for_date(...)

        # --- Broker / Portfolio ---
        portfolio: Any,                 # PortfolioLite-Instanz mit .step(px_t1, w_target, cash_factor)
        initial_cash: float = 1.0,      # Start-NAV der Env

        # --- Risk-free ---
        rf_factor: np.ndarray,          # [T] raw daily_factor (Cash-Accounting im Broker)
        rf_rate: Optional[np.ndarray] = None,  # [T] tägliche Rate als Feature; None => rf_factor - 1

        # --- Reward ---
        reward_kind: str = "log",       # "log" | "icvar" | "icvar_dd"
        reward_scaler: Optional[Any] = None,   # dein reward_norm (online): reset(), update(x)->y
        evaluator: Optional[Any] = None,       # Objekt/Modul mit mts_var_cvar_icvar / _mdd_series
        icvar_alpha: float = 0.05,
        icvar_min_period: int = 1,
        icvar_ewm_alpha: Optional[float] = None,
        icvar_mode: str = "ex_post",    # "ex_post" (inkl. aktueller r) oder "ex_ante"
        lambda_icvar: float = 1.0,      # Gewicht für ICVaR-Penalty
        gamma_dd: float = 0.0,          # Zusatzgewicht für ΔMDD in "icvar_dd"

        # --- Recorder (optional) ---
        recorder: Optional[Any] = None,

        # --- Episodenfenster ---
        start_idx: int = 0,
        end_idx_exclusive: Optional[int] = None,  # EXKLUSIV! step(t) braucht t+1 < end_idx_exclusive

        # --- Sonstiges ---
        validate_actions: bool = False,  # Notfall-Schutz (Policy hat Softmax → meist False)
        eps: float = 1e-12,
    ):
        super().__init__()

        # ---------- Daten & Handles ----------
        self.panel = panel_clean
        self.dates = pd.DatetimeIndex(dates)
        self.assets = list(assets)
        self.spec = spec
        self.SB = state_builder          # hat build_state_for_date(...)
        self.portfolio = portfolio
        self.initial_cash = float(initial_cash)
        self.recorder = recorder

        # Risk-free Zeitreihen
        self.RF_FACTOR = np.asarray(rf_factor, dtype=float).reshape(-1)
        self.RF_RATE = (np.asarray(rf_rate, dtype=float).reshape(-1)
                        if rf_rate is not None else (self.RF_FACTOR - 1.0))

        # Reward-Optionen
        self.reward_kind = str(reward_kind)
        self.reward_scaler = reward_scaler
        self.evaluator = evaluator
        self.icvar_alpha = float(icvar_alpha)
        self.icvar_min_period = int(icvar_min_period)
        self.icvar_ewm_alpha = None if icvar_ewm_alpha is None else float(icvar_ewm_alpha)
        self.icvar_mode = str(icvar_mode)
        self.lambda_icvar = float(lambda_icvar)
        self.gamma_dd = float(gamma_dd)

        self.validate_actions = bool(validate_actions)
        self.eps = float(eps)

        # ---------- Dimensionen ----------
        self.T = len(self.dates)      # #Zeitschritte
        self.A = len(self.assets)     # #Assets
        assert self.RF_FACTOR.shape[0] == self.T, "rf_factor Länge muss T entsprechen."
        if self.RF_RATE.shape[0] != self.T:
            raise ValueError("rf_rate/rf_factor-Längen passen nicht zu 'dates'.")

        # Episoden-Grenzen (t+1 muss existieren => letzter gültiger t = end_excl - 2)
        end_excl = self.T if end_idx_exclusive is None else min(int(end_idx_exclusive), self.T)
        self.last_step = end_excl - 2
        assert self.last_step >= 0, "Zeitfenster zu klein: mindestens 2 Zeitpunkte notwendig."
        self.start_idx = int(start_idx)
        assert 0 <= self.start_idx <= self.last_step, "start_idx außerhalb gültigen Bereichs."

        # ---------- Probe-State → Observation-Space ----------
        # Start-Portfolio: 100% Cash (Assets=0)
        w0_assets = pd.Series(0.0, index=self.assets)
        rf0 = float(self.RF_RATE[self.start_idx])
        probe = self.SB.build_state_for_date(
            panel_clean=self.panel,
            date=self.dates[self.start_idx],
            spec=self.spec,
            assets_order=self.assets,
            portfolio_snapshot={"weights": w0_assets, "cash": self.initial_cash},
            riskfree=rf0,
        )
        X_shape = tuple(np.asarray(probe["X"]).shape)                 # [C,H,W] o.ä.
        g_scalars_shape = tuple(np.asarray(probe["g_scalars"]).shape) # [G]
        g_weights_shape = tuple(np.asarray(probe["g_weights"]).shape) # [A]
        position_shape = (self.A + 1,)                                # (A+1) inkl. Cash

        self.observation_space = spaces.Dict({
            "X":         spaces.Box(-np.inf, np.inf, shape=X_shape,         dtype=np.float32),
            "g_scalars": spaces.Box(-np.inf, np.inf, shape=g_scalars_shape, dtype=np.float32),
            "g_weights": spaces.Box(0.0,     1.0,     shape=g_weights_shape, dtype=np.float32),
            "position":  spaces.Box(0.0,     1.0,     shape=position_shape,  dtype=np.float32),
        })
        self.action_space = spaces.Box(0.0, 1.0, shape=(self.A + 1,), dtype=np.float32)

        # ---------- Laufzustand ----------
        self.t: Optional[int] = None
        self.value: float = float(self.initial_cash)
        self.cash: float = float(self.initial_cash)
        self.equity: float = 0.0
        self.w_prev_full: np.ndarray = np.r_[np.zeros(self.A), 1.0]  # (A+1) 100% Cash
        self.r_log_hist: list[float] = []
        self.nav_hist: list[float] = []

    # --------------------- Gymnasium API ---------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.t = int(options.get("start_idx", self.start_idx)) if options else self.start_idx

        # Broker ggf. zurücksetzen
        if hasattr(self.portfolio, "reset"):
            self.portfolio.reset(initial_cash=self.initial_cash)

        # Online-Reward-Normalizer resetten (falls zustandsbehaftet)
        if self.reward_scaler is not None and hasattr(self.reward_scaler, "reset"):
            self.reward_scaler.reset()

        # Anfangszustand
        self.value = float(self.initial_cash)
        self.cash = float(self.initial_cash)
        self.equity = 0.0
        self.w_prev_full = np.r_[np.zeros(self.A), 1.0]
        self.r_log_hist.clear()
        self.nav_hist = [self.value]

        obs = self._build_obs_at(self.t)

        # --- Baseline: Round 0 Snapshot beim Reset (für Evaluator-Start) ---
        if getattr(self, "recorder", None) is not None:
            import pandas as pd
            date_t = self.dates[self.t]  # aktuelles t nach reset()
            px_t = self.panel.xs(date_t, level=0)

            # Mark-Preis für Bewertung (wie im Broker/Evaluator)
            col_mark = getattr(self.portfolio, "col_mark", "adj_close")
            if col_mark in px_t.columns:
                p0 = px_t[col_mark]
            elif "adj_close" in px_t.columns:
                p0 = px_t["adj_close"]
            elif "close" in px_t.columns:
                p0 = px_t["close"]
            else:
                p0 = pd.Series(index=self.assets, dtype=float)

            # Start: keine Trades/Fees; Shares meist 0; w_post = 0 (nur Asset-Gewichte!)
            shares0 = getattr(self.portfolio, "shares", pd.Series(0.0, index=self.assets))
            zeros_exec = pd.DataFrame(
                0.0, index=self.assets,
                columns=["q", "p_ref", "p_exec", "notional_abs", "spread_cost"]
            )
            zeros_fees = pd.DataFrame(
                0.0, index=self.assets,
                columns=["spread_cost", "fees", "vol_slip", "total_cost"]
            )

            self.recorder.log_round(
                t=date_t,
                assets=self.assets,
                p1=p0,
                cash=float(self.cash),
                shares=shares0,
                w_post=pd.Series(0.0, index=self.assets),
                exec_df=zeros_exec,
                fees_df=zeros_fees,
                round_id=0,
            )
        # --- Ende Baseline ---

        info = self._info(done_reason="reset")
        return obs, info

    def step(self, action: np.ndarray):
        assert self.t is not None, "reset() zuerst aufrufen."
        # Kein t+1 mehr verfügbar → truncaten
        if self.t > self.last_step:
            return self._build_obs_at(self.t), 0.0, False, True, self._info(done_reason="truncated")

        # 1) Action: Zielgewichte inkl. Cash (A+1, Softmax der Policy)
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] != self.A + 1:
            raise ValueError(f"Action-Länge {a.shape[0]} ungleich A+1={self.A+1}.")
        if getattr(self, "validate_actions", False):
            a = np.clip(a, 0.0, None)
            s = a.sum()
            a = (np.r_[np.zeros(self.A), 1.0] if s <= self.eps else a / s)
        w_target_assets = pd.Series(a[:self.A], index=self.assets)  # Broker erwartet Assets-only

        # 2) Preise/RF @ t+1
        date_t1 = self.dates[self.t + 1]
        px_t1 = self.panel.xs(date_t1, level=0)             # DataFrame mit adj_open/adj_close/...
        cash_factor_t1 = float(self.RF_FACTOR[self.t + 1])   # raw daily_factor fürs Cash

        # 3) Broker/PortfolioLite: Ausführung & Kosten (Turnover/Fees/Spread IM BROKER)
        # >>> Anpassen, falls deine Signatur anders ist:
        w_series_post, info = self.portfolio.step(
            px_t1=px_t1,
            w_target=w_target_assets,
            cash_factor=cash_factor_t1,
        )
        # Gewichte vom Broker (Assets-only ODER Assets+Cash) robust aufbereiten
        w_assets_post, cash_weight_post, w_full_post = self._unpack_weights(w_series_post)

        # Portfoliowerte aus Broker-Info
        value_t = float(self.value)
        value_t1 = float(info.get("value", np.nan))
        self.cash = float(info.get("cash", np.nan))
        self.equity = value_t1 - self.cash

        # 4) Guards
        if not np.isfinite(value_t1) or not np.isfinite(value_t):
            obs = self._build_obs_at(self.t)
            return obs, -1.0, True, False, self._info(done_reason="invalid_value")
        ret_raw = value_t1 / max(value_t, self.eps)
        if (ret_raw <= 0.0) or (not np.isfinite(ret_raw)):
            obs = self._build_obs_at(self.t)
            return obs, -1.0, True, False, self._info(done_reason="nonpositive_ret")
        r_log = float(np.log(ret_raw))

        # 5) Reward (deine Evaluator-Funktionen)
        include_current = (self.icvar_mode.lower() == "ex_post")
        # Serie für ICVaR: Historie (+ optional aktueller r_log)
        r_series = pd.Series(self.r_log_hist + ([r_log] if include_current else []))

        if self.reward_kind == "log":
            r_raw = r_log

        elif self.reward_kind in ("icvar", "icvar_dd"):
            if self.evaluator is None or not hasattr(self.evaluator, "mts_var_cvar_icvar"):
                raise RuntimeError("Evaluator mit mts_var_cvar_icvar(...) nicht gesetzt.")
            # ICVaR_t (Loss-Skala) via DEINE Funktion
            var_s, cvar_s, icvar_s = self.evaluator.mts_var_cvar_icvar(
                r_series,
                alpha=self.icvar_alpha,
                min_period=self.icvar_min_period,
                include_current=include_current,
                ewm_alpha=self.icvar_ewm_alpha,
                as_series=True,
            )
            icvar_t = float(icvar_s.iloc[-1]) if len(icvar_s) else 0.0
            penalty_icvar = self.lambda_icvar * icvar_t

            if self.reward_kind == "icvar":
                r_raw = r_log - penalty_icvar
            else:
                # ΔMDD über NAV-Pfad (ohne Zukunftsleak)
                if self.evaluator is None or not hasattr(self.evaluator, "_mdd_series"):
                    raise RuntimeError("Evaluator mit _mdd_series(...) nicht gesetzt.")
                nav_t_path = pd.Series(self.nav_hist)  # bis NAV_t
                nav_t1_path = pd.concat([nav_t_path, pd.Series([value_t1])], ignore_index=True)
                mdd_t = float(self.evaluator._mdd_series(nav_t_path.ffill()).iloc[-1]) if len(nav_t_path) else 0.0
                mdd_t1 = float(self.evaluator._mdd_series(nav_t1_path.ffill()).iloc[-1]) if len(nav_t1_path) else 0.0
                delta_mdd = max(0.0, mdd_t1 - mdd_t)
                r_raw = r_log - penalty_icvar - self.gamma_dd * delta_mdd
        else:
            raise ValueError(f"Unbekannte reward_kind: {self.reward_kind}")

        reward = r_raw if (self.reward_scaler is None) else float(self.reward_scaler.update(r_raw))

        # 6) Recorder (optional)
        if self.recorder is not None:
            try:
                close_t1 = px_t1.get(getattr(self.portfolio, "col_mark", "adj_close"),
                                     pd.Series(index=self.assets, dtype=float))
                shares_t1 = getattr(self.portfolio, "shares", pd.Series(0.0, index=self.assets))
                exec_df = info.get("trades")
                fees_df = info.get("fees_detail")
                self.recorder.log_round(
                    t=date_t1, assets=self.assets, p1=close_t1, cash=self.cash,
                    shares=shares_t1, w_post=w_assets_post,
                    exec_df=exec_df, fees_df=fees_df,
                    round_id=int(self.t + 1)
                )
            except Exception:
                # Recorder ist optional; Env soll robust bleiben
                pass

        # 7) Zustand fortschreiben & Historien aktualisieren (KAUSAL NACH Reward)
        self.t += 1
        self.value = value_t1
        self.w_prev_full = w_full_post
        self.r_log_hist.append(r_log)
        self.nav_hist.append(value_t1)

        # 8) Next-Obs & Done
        obs_next = self._build_obs_at(self.t)
        terminated = bool(self.value <= 0.0)       # Insolvenz
        truncated = bool(self.t > self.last_step)  # kein weiteres t+1

        info_out = self._info(
            fees=float(info.get("fees", 0.0)),
            turnover=float(info.get("controls", {}).get("l1_step", np.nan)),
            ret_raw=float(ret_raw),
            r_log=float(r_log),
            reward=float(reward),
            cash_weight=float(cash_weight_post),
            trades=info.get("trades"),
            fees_detail=info.get("fees_detail"),
            controls=info.get("controls"),
            exec_date=date_t1,
        )
        return obs_next, reward, terminated, truncated, info_out

    # --------------------- Hilfsfunktionen ---------------------
    def _unpack_weights(self, w_series: pd.Series) -> tuple[pd.Series, float, np.ndarray]:
        """
        Nimmt eine pd.Series mit Gewichten vom Broker.
        Akzeptiert entweder:
          - nur Asset-Gewichte (Cash fehlt), oder
          - Asset-Gewichte + Cash (Index 'cash'/'__cash__'/'_cash', case-insensitive).
        Gibt zurück:
          (w_assets: Series in self.assets-Reihenfolge, cash_w: float, w_full: np.ndarray [A+1])
        """
        s = w_series.copy()
        s.index = s.index.astype(str)
        w_assets = s.reindex(self.assets).fillna(0.0).astype(float)
        cash_keys = [k for k in s.index if k.strip().lower() in ("cash", "__cash__", "_cash")]
        if cash_keys:
            cash_w = float(s[cash_keys[0]])
        else:
            cash_w = float(max(0.0, 1.0 - w_assets.sum()))
        total = float(w_assets.sum() + cash_w)
        if total > 0:
            w_assets = (w_assets / total).astype(float)
            cash_w = float(cash_w / total)
        w_full = np.r_[w_assets.values, cash_w]
        return w_assets, cash_w, w_full

    def _build_obs_at(self, t: int) -> Dict[str, np.ndarray]:
        """Baut die Observation für Datum t via state_builder; hängt 'position' (A+1 inkl. Cash) an."""
        w_assets = pd.Series(self.w_prev_full[:self.A], index=self.assets)
        rf_rate_t = float(self.RF_RATE[t])
        s = self.SB.build_state_for_date(
            panel_clean=self.panel,
            date=self.dates[t],
            spec=self.spec,
            assets_order=self.assets,
            portfolio_snapshot={"weights": w_assets, "cash": float(self.cash)},
            riskfree=rf_rate_t,
        )
        obs = {
            "X":         np.asarray(s["X"], dtype=np.float32),
            "g_scalars": np.asarray(s["g_scalars"], dtype=np.float32),
            "g_weights": np.asarray(s["g_weights"], dtype=np.float32),
            "position":  np.asarray(self.w_prev_full, dtype=np.float32),
        }
        return obs

    def _info(self, done_reason: Optional[str] = None, **extras) -> Dict[str, Any]:
        """Reichhaltiges Info-Dict (für Debug/Plots/Trainer-Logging)."""
        date = self.dates[int(self.t)] if (self.t is not None and 0 <= self.t < len(self.dates)) else None
        info = {
            "date": date,
            "step": int(self.t) if self.t is not None else int(self.start_idx),
            "value": float(self.value),
            "cash": float(self.cash),
            "equity": float(self.value - self.cash),
            "weights_prev": self.w_prev_full.copy(),
        }
        if done_reason is not None:
            info["done_reason"] = done_reason
        for k, v in extras.items():
            info[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        return info
