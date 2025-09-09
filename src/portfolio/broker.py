import numpy as np
import pandas as pd

from src.portfolio import execution as _execution
from src.portfolio import fees as _fees
from src.portfolio.execution_controls import apply_execution_controls
from src.portfolio.vol_controls import VolEWMA, vol_target_step

EPS = 1e-12

class PortfolioLite:
    def __init__(self, assets, initial_cash=1_000_000.0,
                 col_mark='adj_close', col_ref='adj_open',
                 col_spread='bid_ask_spread_corwin_schultz',
                 lot_size=1, fee_kwargs=None,
                 execution_mod=None, fees_mod=None,
                 use_vol_targeting=False,
                 target_vol_annual=0.10,  # 10% p.a. (nur falls vol-targeting an)
                 vol_span=60,  # EWMA-Fenster (daily)
                 min_change_l1=0.03,  # No-Trade-Band (L1)
                 max_step_l1=0.3):  # Turnover-Cap (L1)
        self.assets = list(assets); self.A = len(self.assets)
        self.col_mark, self.col_ref, self.col_spread = col_mark, col_ref, col_spread
        self.lot_size =  int(lot_size)
        self.exec = execution_mod or _execution
        self.fees = fees_mod or _fees
        self.fee_kwargs = fee_kwargs or {}

        self.use_vol_targeting = bool(use_vol_targeting)
        self.target_vol_daily = float(target_vol_annual) / np.sqrt(252.0)
        self.min_change_l1 = float(min_change_l1)
        self.max_step_l1 = float(max_step_l1)
        self.sigma_hat = 0.0
        self._last_value = float(initial_cash)  # für tägliche r_t
        self.vol = VolEWMA(span=int(vol_span))
        self.reset(initial_cash)

    def reset(self, initial_cash):
        self.cash = float(initial_cash)
        self.shares = pd.Series(0.0, index=self.assets)
        self.value = float(initial_cash)
        self.weights = pd.Series(0.0, index=self.assets)
        span = getattr(self.vol, "span", 60)
        self.vol = VolEWMA(span=span)
        self.sigma_hat = 0.0
        self._last_value = float(initial_cash)

    def step(self, px_t1: pd.DataFrame, w_target: pd.Series, cash_factor=None):

        # 1) Preise @ t+1 (Mark-to-Market nach Ausführung)
        adj_open = px_t1[self.col_ref].astype(float).reindex(self.assets)  # t+1 Adj_Open (Execution/Sizing)
        adj_close = px_t1[self.col_mark].astype(float).reindex(self.assets)  # t+1 Adj_Close (Bewertung)
        assert adj_open.notna().any(), "p_ref (t+1 adj_open) hat nur NaNs – kein Handel möglich."
        assert adj_close.notna().any(), "p_mark (t+1 adj_close) hat nur NaNs – Bewertung nicht möglich."

        if not getattr(self, "_did_openadj_check", False):
            if self.col_ref == "adj_open" and self.col_mark == "adj_close":
                print(f"[broker CHECK] NaNs: open_adj={int(adj_open.isna().sum())}, adj_close={int(adj_close.isna().sum())}")
            self._did_openadj_check = True

        # 2) Zielgewichte vorbereiten (clip/norm)
        w = w_target.reindex(self.assets).fillna(0.0).clip(lower=0.0)

        # 2a) sicherstellen das keine Assets gehandelt werden welche noch nicht am Markt verfügbar sind
        tradable = adj_open.notna().copy()

        attempted_untradable = float(w.where(~tradable, 0.0).sum())
        w = w.where(tradable, 0.0)  # verbieten statt umverteilen

        #-----------------------------------------------------------
        tradable = adj_open.notna()

        attempted_untradable = float(w.where(~tradable, 0.0).sum())  # nur Logging/Stats
        w = w.where(tradable, 0.0)  # verbieten statt umverteilen

        # 1) Keine NaNs/Inf
        assert np.isfinite(w.values).all(), "NaNs/Inf in Zielgewichten nach Maskierung"

        # 2) Long-only invariant (sollte durch clip schon gelten)
        assert (w.values >= -1e-12).all(), "Negative Gewichte nach Maskierung"

        # 3) Summe <= 1 (Rest = Cash); obere Schranke greift unten


        # Budget-Schranke nur nach oben (Rest bleibt Cash)
        budget = float(w.sum())
        if budget > 1.0 + EPS:
            w = w / budget

        # (optional) Log:
        # self.logger.debug(f"masked_untradable={attempted_untradable:.6f}, budget={w.sum():.6f}")

        # Prev-Weights (auf gleiche Reihenfolge wie assets)
        w_prev = self.weights.reindex(self.assets).fillna(0.0)

        # (Optional) Vol-Targeting: nur Schrittweite skalieren, nie über Ziel hinaus
        if self.use_vol_targeting:
            w_after_vol, applied_scale = vol_target_step(
                weights_prev=w_prev.values,
                weights_target=w.values,
                vol_estimate=self.sigma_hat,          # daily
                target_vol=self.target_vol_daily,     # daily
                scaling_limits=(0.5, 2.0)
            )
            # zurück in Series-Form
            w = pd.Series(w_after_vol, index=self.assets)
        # else: w unverändert lassen

        # No-Trade-Band + Turnover-Cap (L1), Reihenfolge: erst NTB, dann Cap
        w_exec, info_ctrl = apply_execution_controls(
            weights_prev=w_prev.values,
            weights_target=w.values,
            min_change_l1=self.min_change_l1,
            max_step_l1=self.max_step_l1,
        )
        w = pd.Series(w_exec, index=self.assets)

        # 3) Portfolio-Wert vor Rebalance zum t+1-Preis
        Ppre = self.cash + float((self.shares * adj_open).sum())

        if not getattr(self, "_did_start_ping", False):
            print(f"[broker PING] A={len(self.assets)}, Ppre={Ppre:,.2f}, cash={self.cash:,.2f}")
            self._did_start_ping = True

        # 3a) Ziel-Stückzahlen @ t+1
        target_shares = (w * Ppre) / adj_open.replace(0.0, np.nan)
        target_shares = target_shares.fillna(0.0)

        # 3b) Delta-Stücke & Lot-Rundung
        q = target_shares - self.shares
        q = self.exec.round_shares(q, lot=self.lot_size)  # Series -> Series

        # 4) Execution (eine Wahrheit: plan_execution_series)
        spread = px_t1.get(self.col_spread, pd.Series(0.0, index=adj_open.index)).astype(float)

        exec_df = self.exec.plan_execution_series(
            q=q, adj_open=adj_open, spread=spread,
        )
        # exec_df Spalten: ["q","p_ref","p_exec","notional_abs","spread_cost"]

        # 4b) Fees
        fees_df = self.fees.apply_fees(exec_df, **self.fee_kwargs)
        fees_total = float(fees_df["total_cost"].sum())
        cash_delta = float((exec_df["q"] * exec_df["p_exec"]).sum()) + fees_total

        # --- Cash-Guard: falls Cash negativ würde, q skalieren und neu rechnen
        if self.cash - cash_delta < 0.0 and cash_delta > EPS:
            eta = max(0.0, min(1.0, self.cash / cash_delta))  # Anteil finanzierbar
            q = self.exec.round_shares(q * eta, lot=self.lot_size)
            q = q.where(self.shares + q >= 0.0, -self.shares)
            exec_df = self.exec.plan_execution_series(q=q, adj_open=adj_open, spread=spread)
            fees_df = self.fees.apply_fees(exec_df, **self.fee_kwargs)
            cash_delta = float((exec_df["q"] * exec_df["p_exec"]).sum()) + float(fees_df["total_cost"].sum())

        # --- Cash-Guard (2): nach Rundung/Fees sicherstellen, dass Cash >= 0
        if self.cash - cash_delta < 0.0 and cash_delta > EPS:
            # Greedy: kleinste Kauf-Tickets streichen, bis der Cash reicht
            buys = exec_df[exec_df["q"] > 0].copy()
            if not buys.empty:
                # Ticketkosten inkl. Fees für Sortierung
                ticket_cost = buys["q"] * buys["p_exec"] + fees_df.loc[buys.index, "total_cost"]
                for idx in ticket_cost.sort_values().index:
                    # Ticket entfernen
                    q.loc[idx] = 0.0
                    # Execution + Fees neu rechnen
                    exec_df = self.exec.plan_execution_series(q=q, adj_open=adj_open, spread=spread)
                    fees_df = self.fees.apply_fees(exec_df, **self.fee_kwargs)
                    cash_delta = float((exec_df["q"] * exec_df["p_exec"]).sum()) + float(fees_df["total_cost"].sum())
                    if self.cash - cash_delta >= 0.0:
                        break

            # Falls immer noch knapp negativ (numerisch): alles kaufen stoppen
            if self.cash - cash_delta < 0.0:
                q = q.where(q <= 0.0, 0.0)
                exec_df = self.exec.plan_execution_series(q=q, adj_open=adj_open, spread=spread)
                fees_df = self.fees.apply_fees(exec_df, **self.fee_kwargs)
                cash_delta = float((exec_df["q"] * exec_df["p_exec"]).sum()) + float(fees_df["total_cost"].sum())

        # 5) State-Update (Cash, Shares, Value, Weights)
        self.cash = self.cash - cash_delta
        # Cash-Zins (Schritt t->t1) anwenden, falls übergeben
        if cash_factor is not None:
            self.cash *= float(cash_factor)


        self.shares = (self.shares + exec_df["q"]).reindex(self.assets).fillna(0.0)
        self.value = self.cash + float((self.shares * adj_close).sum())
        self.weights = (self.shares * adj_close) / max(self.value, EPS)
        w_cash_post = max(0.0, min(1.0, float(self.cash) / max(self.value, EPS)))
        weights_with_cash = pd.concat([self.weights, pd.Series({"CASH": w_cash_post})])

        # --- Vol-Schätzer updaten (realized daily return, netto)
        try:
            r_t = (self.value / max(self._last_value, EPS)) - 1.0
        except Exception:
            r_t = 0.0
        self.sigma_hat = self.vol.update(r_t)
        self._last_value = self.value


        # 6) Info für Debug/Analyse
        info = {
            "value": self.value,
            "cash": self.cash,
            "fees": fees_total,
            "q": exec_df["q"],
            "pexec": exec_df["p_exec"],
            "trades": exec_df,  # konsistent: das sind die Trades
            "fees_detail": fees_df[["spread_cost", "fees", "vol_slip", "total_cost"]],
            "Ppre_open": Ppre,
            "w_open_pre": ((self.shares - exec_df["q"]) * adj_open) / max(Ppre, EPS),  # Gewichte direkt vor Ausführung
            "w_target": w,
            "w_close_post_with_cash": weights_with_cash,  # Series: [A+1]
            "attempted_untradable_weight" : attempted_untradable,
            "controls": info_ctrl,  # acted, l1_distance, applied_scale, l1_step, ...
            "sigma_hat_daily": float(self.sigma_hat),
            "vol_targeting": bool(self.use_vol_targeting),
        }
        self.weights_with_cash = weights_with_cash



        return self.weights_with_cash, info  # Reward berechnet die Env/der Loop

