import numpy as np
import gymnasium as gym
from gymnasium import spaces

def _project_to_simplex(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Projiziere auf den Simplex {w >= 0, sum w = 1} (Condat/Michelot)."""
    v = np.maximum(w, 0.0).astype(float)
    if v.sum() <= 1.0 + eps:
        s = v.sum()
        if s <= eps:
            # Fallback: alles Cash
            out = np.zeros_like(v)
            out[-1] = 1.0
            return out
        return v / s
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w_hat = np.maximum(v - theta, 0.0)
    s = w_hat.sum()
    return w_hat / (s if s > eps else 1.0)

class TradingEnv(gym.Env):
    """
    Gymnasium-kompatibles Trading-Env (Dict-Obs, T+1-Ausführung).
    - Execution: adj_open[t+1] (Spread/Fees im Broker)
    - Bewertung/Value: via Broker (der adj_close[t+1] nutzt)
    - Action: Zielgewichte (A+1 inkl. Cash), long-only Simplex
    - Reward: log(value_{t+1}/value_t) nach Kosten, online-normalisiert (reward_norm)
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        *,
        state_builder,                 # muss: build_obs(t:int, position:np.ndarray) -> {"panel","global","position"}
        broker,                        # muss: execute(...) oder step(...), siehe unten
        open_px: np.ndarray,           # Shape [T, A] (adj_open)
        close_px: np.ndarray,          # Shape [T, A] (adj_close) (vom Broker genutzt)
        rf_factor: np.ndarray,         # Shape [T] (raw daily_factor), Index-align zu Preisen
        reward_scaler=None,            # Objekt mit reset(), update(x)->y  (z.B. deine reward_norm)
        dates: np.ndarray | None = None,  # optional: Index für info["date"]
        start_idx: int = 0,
        end_idx: int | None = None,    # exklusiv; default = T-1 (damit t+1 existiert bis end_idx-1)
        action_clip_eps: float = 1e-9,
    ):
        super().__init__()
        self.state_builder = state_builder
        self.broker = broker
        self.OPEN = np.asarray(open_px)
        self.CLOSE = np.asarray(close_px)
        self.RF = np.asarray(rf_factor).reshape(-1)
        self.dates = np.asarray(dates) if dates is not None else None
        self.A = self.OPEN.shape[1]                              # Anzahl Assets (ohne Cash)
        self.T = self.OPEN.shape[0]
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx) if end_idx is not None else (self.T - 1)  # letzter gültiger t (da wir t+1 brauchen)
        self.action_clip_eps = float(action_clip_eps)

        # Initiale Gewichte (100% Cash).
        self.w0 = np.zeros(self.A + 1, dtype=float)
        self.w0[-1] = 1.0

        # Reward-Scaler (online). Falls None: Identität.
        self.reward_scaler = reward_scaler

        # Observation/Action spaces (Shapes via Probe-State bestimmen)
        probe_obs = self.state_builder.build_obs(self.start_idx, position=self.w0.copy())
        assert isinstance(probe_obs, dict) and "panel" in probe_obs and "global" in probe_obs and "position" in probe_obs, \
            "state_builder.build_obs() muss Dict mit Keys {'panel','global','position'} liefern."

        panel_shape = tuple(np.asarray(probe_obs["panel"]).shape)    # (W, A, F)
        global_shape = tuple(np.asarray(probe_obs["global"]).shape)  # (G,)
        position_shape = (self.A + 1,)                               # (A+1,)

        self.observation_space = spaces.Dict({
            "panel":    spaces.Box(low=-np.inf, high=np.inf, shape=panel_shape, dtype=np.float32),
            "global":   spaces.Box(low=-np.inf, high=np.inf, shape=global_shape, dtype=np.float32),
            "position": spaces.Box(low=0.0,     high=1.0,     shape=position_shape, dtype=np.float32),
        })
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.A + 1,), dtype=np.float32)

        # Laufvariablen
        self.t = None
        self.value = None
        self.w_prev = None
        self.cash = None
        self.equity = None

    # -- Gymnasium API --

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.reward_scaler is not None and hasattr(self.reward_scaler, "reset"):
            self.reward_scaler.reset()

        # Startindex ggf. aus options übersteuern
        if options and "start_idx" in options:
            self.t = int(options["start_idx"])
        else:
            self.t = self.start_idx

        # Startzustand
        self.value = 1.0
        self.w_prev = self.w0.copy()
        self.cash = self.value  # 100% Cash
        self.equity = 0.0

        obs = self._build_obs(self.t, self.w_prev)
        info = self._info(done_reason="reset")
        return obs, info

    def step(self, action: np.ndarray):
        # --- 1) Action -> Zielgewichte (Simplex, >=0, Sum=1) ---
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] != self.A + 1:
            raise ValueError(f"Action shape {a.shape} ungleich erwarteter Länge A+1={self.A+1}.")
        a = np.clip(a, 0.0, 1.0)
        w_target = _project_to_simplex(a, eps=self.action_clip_eps)

        # --- 3) Execution auf t+1 (Broker) ---
        if self.t >= self.end_idx:
            # Es gibt kein t+1 mehr → sofort truncaten
            obs = self._build_obs(self.t, self.w_prev)
            return obs, 0.0, False, True, self._info(done_reason="truncated_before_step")

        open_tp1 = self.OPEN[self.t + 1, :]
        close_tp1 = self.CLOSE[self.t + 1, :]
        rf_tp1 = float(self.RF[self.t + 1])

        # Broker/PortfolioLite aufrufen (Spread/Fees dort implementiert)
        exec_fn = getattr(self.broker, "execute", None) or getattr(self.broker, "step", None)
        if exec_fn is None:
            raise AttributeError("Broker hat weder .execute(...) noch .step(...).")

        w_exec, cash_tp1, equity_tp1, value_tp1, fees_t, turnover_t = exec_fn(
            w_prev=self.w_prev,
            open_tp1=open_tp1,
            close_tp1=close_tp1,
            rf_factor_tp1=rf_tp1,
        )

        # --- 4) Reward (log nach Kosten) ---
        value_t = float(self.value)
        value_tp1 = float(value_tp1)
        if value_t <= 0.0 or np.isnan(value_t) or np.isnan(value_tp1) or value_tp1 < 0.0:
            reward_log = -np.inf
            reward = -1.0 if self.reward_scaler is None else self.reward_scaler.update(-1.0)
            terminated = True
            truncated = False
            info = self._info(
                done_reason="invalid_value",
                fees=float(fees_t),
                turnover=float(turnover_t),
                ret_raw=np.nan,
                r_log=np.nan,
            )
            obs = self._build_obs(self.t, self.w_prev)  # irgendein obs zurück
            return obs, float(reward), terminated, truncated, info

        ret_raw = value_tp1 / max(value_t, 1e-16)
        r_log = np.log(ret_raw)
        reward = r_log if self.reward_scaler is None else float(self.reward_scaler.update(r_log))

        # --- 5) Fortschreiben t -> t+1 ---
        self.t += 1
        self.w_prev = np.asarray(w_exec, dtype=float)
        self.value = value_tp1
        self.cash = float(cash_tp1)
        self.equity = float(equity_tp1)

        # --- 6) Next-Obs ---
        obs_next = self._build_obs(self.t, self.w_prev)

        # --- 7) Done-Signale ---
        terminated = bool(self.value == 0.0)
        truncated = bool(self.t >= self.end_idx)

        info = self._info(
            fees=float(fees_t),
            turnover=float(turnover_t),
            ret_raw=float(ret_raw),
            r_log=float(r_log),
        )
        return obs_next, float(reward), terminated, truncated, info

    # -- Hilfen --

    def _build_obs(self, t: int, position: np.ndarray) -> dict:
        obs = self.state_builder.build_obs(t, position=np.asarray(position, dtype=float))
        # Dtypes angleichen
        return {
            "panel":    np.asarray(obs["panel"], dtype=np.float32),
            "global":   np.asarray(obs["global"], dtype=np.float32),
            "position": np.asarray(obs["position"], dtype=np.float32),
        }

    def _info(self, done_reason: str | None = None, **extras) -> dict:
        date = None
        if self.dates is not None and self.t is not None and 0 <= self.t < len(self.dates):
            date = self.dates[self.t]
        base = {
            "date": date,
            "step": int(self.t if self.t is not None else self.start_idx),
            "weights_prev": self.w_prev.copy() if self.w_prev is not None else None,
            "cash": float(self.cash if self.cash is not None else 0.0),
            "equity": float(self.equity if self.equity is not None else 0.0),
            "value": float(self.value if self.value is not None else 1.0),
        }
        if done_reason is not None:
            base["done_reason"] = done_reason
        base.update({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in extras.items()})
        return base
