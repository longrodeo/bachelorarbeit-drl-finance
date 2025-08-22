import numpy as np
import pandas as pd

EPS = 1e-12

class PortfolioLite:
    def __init__(self, assets, initial_cash=1_000_000.0,
                 col_mark='close', col_ref='exec_ref_tplus1', col_spread='spread_cs',
                 allow_short=False, lot_size=1, fee_kwargs=None,
                 execution_mod=None, fees_mod=None):
        self.assets = list(assets); self.A = len(self.assets)
        self.col_mark, self.col_ref, self.col_spread = col_mark, col_ref, col_spread
        self.allow_short, self.lot_size = allow_short, int(lot_size)
        self.exec, self.fees = execution_mod, fees_mod
        self.fee_kwargs = fee_kwargs or {}
        self.reset(initial_cash)

    def reset(self, initial_cash):
        self.cash = float(initial_cash)
        self.shares = pd.Series(0.0, index=self.assets)
        self.value = float(initial_cash)
        self.weights = pd.Series(0.0, index=self.assets)

    def step(self, px_t: pd.Series, px_t1: pd.DataFrame, w_target: pd.Series):
        # 1) Werte vor Rebalance
        P_t  = self.cash + float((self.shares * px_t).sum())
        p1   = px_t1[self.col_mark].astype(float)
        Ppre = self.cash + float((self.shares * p1).sum())

        # 2) Zielgewichte (clip/norm)
        w = w_target.clip(lower=0.0) if not self.allow_short else w_target.copy()
        w = w / max(w.sum(), EPS)

        # 3) Ziel-Stücke @ t+1 mark  (ohne Rundung/Budget)
        target_shares = (w * Ppre) / p1.replace(0.0, np.nan)
        target_shares = target_shares.fillna(0.0)
        q = (target_shares - self.shares)

        # 4) Execution-Preise (Half-Spread) & Fees
        pref   = px_t1[self.col_ref].astype(float)
        spread = px_t1.get(self.col_spread, pd.Series(0.0, index=self.assets)).astype(float)
        pexec = pd.Series(pref, index=self.assets, dtype=float)
        pexec[q > 0]  = self.exec.half_spread_price(pref[q > 0],  +1, spread[q > 0])
        pexec[q < 0]  = self.exec.half_spread_price(pref[q < 0],  -1, spread[q < 0])

        trades = pd.DataFrame({'q': q, 'p_exec': pexec, 'p_ref': pref, 'spread': spread})
        trades['notional_abs'] = (q.abs() * pexec)
        trades['spread_cost']  = (q.abs() * pref * 0.5 * spread)
        fees_total = float(self.fees.apply_fees(trades, **self.fee_kwargs))

        # 5) State-Update
        self.cash   = self.cash - float((q * pexec).sum()) - fees_total
        self.shares = self.shares + q
        self.value  = self.cash + float((self.shares * p1).sum())
        self.weights = (self.shares * p1) / max(self.value, EPS)

        info = {'value': self.value, 'cash': self.cash, 'fees': fees_total,
                'q': q, 'pexec': pexec, 'weights_post': self.weights}
        return self.weights.copy(), info  # kein Reward
