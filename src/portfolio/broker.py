import numpy as np
import pandas as pd

from portfolio import execution as _execution
from portfolio import fees as _fees



EPS = 1e-12

class PortfolioLite:
    def __init__(self, assets, initial_cash=1_000_000.0,
                 col_mark='adj_close', col_ref='open',
                 col_spread='bid_ask_spread_corwin_schultz',
                 lot_size=1, fee_kwargs=None,
                 execution_mod=None, fees_mod=None):
        self.assets = list(assets); self.A = len(self.assets)
        self.col_mark, self.col_ref, self.col_spread = col_mark, col_ref, col_spread
        self.lot_size =  int(lot_size)
        self.exec = execution_mod or _execution
        self.fees = fees_mod or _fees
        self.fee_kwargs = fee_kwargs or {}
        self.reset(initial_cash)

    def reset(self, initial_cash):
        self.cash = float(initial_cash)
        self.shares = pd.Series(0.0, index=self.assets)
        self.value = float(initial_cash)
        self.weights = pd.Series(0.0, index=self.assets)

    def step(self, px_t1: pd.DataFrame, w_target: pd.Series, cash_factor=None):
        # 1) Preise @ t+1 (Mark-to-Market nach Ausführung)
        p_ref = px_t1[self.col_ref].astype(float).reindex(self.assets)  # t+1 Open (Execution/Sizing)
        p_mark = px_t1[self.col_mark].astype(float).reindex(self.assets)  # t+1 Close (Bewertung)
        assert p_ref.notna().any(), "p_ref (t+1 open) hat nur NaNs – kein Handel möglich."
        assert p_mark.notna().any(), "p_mark (t+1 adj_close) hat nur NaNs – Bewertung nicht möglich."

        # 2) Zielgewichte vorbereiten (clip/norm)
        w = w_target.reindex(self.assets).fillna(0.0).clip(lower=0.0)

        # 2a) sicherstellen das keine Assets gehandelt werden welche noch nicht am Markt verfügbar sind
        tradable = p_ref.notna().copy()
        if "CASH" in tradable.index:
            tradable.loc["CASH"] = True

        attempted_untradable = float(w.where(~tradable, 0.0).sum())
        w = w.where(tradable, 0.0)  # verbieten statt umverteilen

        # Budget-Schranke nur nach oben (Rest bleibt Cash)
        budget = float(w.sum())
        if budget > 1.0 + EPS:
            w = w / budget

        # 3) Portfolio-Wert vor Rebalance zum t+1-Preis
        Ppre = self.cash + float((self.shares * p_ref).sum())

        # 3a) Ziel-Stückzahlen @ t+1
        target_shares = (w * Ppre) / p_ref.replace(0.0, np.nan)
        target_shares = target_shares.fillna(0.0)

        # 3b) Delta-Stücke & Lot-Rundung
        q = target_shares - self.shares
        q = self.exec.round_shares(q, lot=self.lot_size)  # Series -> Series

        # 4) Execution (eine Wahrheit: plan_execution_series)
        spread = px_t1.get(self.col_spread, pd.Series(0.0, index=p_ref.index)).astype(float)

        exec_df = self.exec.plan_execution_series(
            q=q, p_ref=p_ref, spread=spread,
            cash_assets={"CASH"},
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
            exec_df = self.exec.plan_execution_series(q=q, p_ref=p_ref, spread=spread, cash_assets={"CASH"})
            fees_df = self.fees.apply_fees(exec_df, **self.fee_kwargs)
            cash_delta = float((exec_df["q"] * exec_df["p_exec"]).sum()) + float(fees_df["total_cost"].sum())

        # 5) State-Update (Cash, Shares, Value, Weights)
        self.cash = self.cash - cash_delta
        # Cash-Zins (Schritt t->t1) anwenden, falls übergeben
        if cash_factor is not None:
            self.cash *= float(cash_factor)


        self.shares = (self.shares + exec_df["q"]).reindex(self.assets).fillna(0.0)
        self.value = self.cash + float((self.shares * p_mark).sum())
        self.weights = (self.shares * p_mark) / max(self.value, EPS)

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
            "w_open_pre": ((self.shares - exec_df["q"]) * p_ref) / max(Ppre, EPS),  # Gewichte direkt vor Ausführung
            "w_target": w,
            "attempted_untradable_weight" : attempted_untradable
        }



        return self.weights.copy(), info  # Reward berechnet die Env/der Loop

