import gymnasium as gym
import numpy as np

class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, broker, state_builder, reward_fn, obs_norm=None,
                 use_vol_targeting=False, min_change_l1=0.03, max_step_l1=0.20, seed=42):
        self.broker = broker
        self.state_builder = state_builder     # liefert Features zum t
        self.reward_fn = reward_fn
        self.obs_norm = obs_norm               # z.B. rolling z-score Wrapper
        self.use_vol_targeting = use_vol_targeting
        self.min_change_l1 = min_change_l1
        self.max_step_l1 = max_step_l1
        self.rng = np.random.default_rng(seed)

        n_assets = broker.n_assets
        n_feats  = state_builder.obs_dim
        self.observation_space = gym.spaces.Box(-10, 10, shape=(n_feats,), dtype=np.float32)
        self.action_space      = gym.spaces.Box(-np.inf, np.inf, shape=(n_assets,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.broker.reset(self.broker.initial_cash)
        obs = self.state_builder.build()
        if self.obs_norm is not None:
            obs = self.obs_norm.transform(obs)
        return obs.astype(np.float32), {}

    def step(self, action):
        from env.action_mapping import action_to_weights_softmax
        w_target = action_to_weights_softmax(action)
        # PortfolioLite nutzt intern: (optional) vol_target_step → NTB → Cap → T+1
        pnl, info_b = self.broker.rebalance_to(w_target)   # du hast bereits die T+1-Logik
        obs = self.state_builder.build()                   # Zustand bei t+1
        if self.obs_norm is not None:
            obs = self.obs_norm.transform(obs)
        reward = self.reward_fn(pnl, info_b)
        terminated = info_b.get("done", False)
        return obs.astype(np.float32), float(reward), bool(terminated), False, info_b
