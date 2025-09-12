# probe_runtime.py
# Minimaler Laufzeit-Probe-Runner für PPO/SAC auf deiner TradingEnv.
# Misst (1) Trainingsdauer bis StopAfterNSteps und (2) Testdauer für N Steps/Episode.
from __future__ import annotations
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Projekt-Imports (unverändert aus deinem Code-Basisstand)
from src.rl.cnn_extractor import CNN1DExtractor
from src.utils.paths import CLEAN_PANEL, FEATURES_NORM, get_asset_groups, get_assets_flat
from src.utils.parquet_io import load_parquet
from src.state.state_builder import load_spec, build_state_for_date
from src.env.trading_env import TradingEnv
from src.portfolio.broker import PortfolioLite
from src.rl.wrapper import ActionMappingWrapper
from src.utils.helpers import set_seed
from src.utils.paths import SPEC_S0_YAML as spec_features

# --- Kleine Profiling-Callbacks ----------------------------------------------
class StopAfterNSteps(BaseCallback):
    def __init__(self, max_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_timesteps = int(max_timesteps)
    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n >= self.max_timesteps:
            if self.verbose:
                print(f"[PROFILE] Stop nach {n:,} Steps")
            return False
        return True

# --- Threads/Deterministik konservativ ---------------------------------------
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# --- Env-Factory (ohne Accounting) -------------------------------------------
def make_env(seed: int = 42, initial_cash: float = 1_000_000.0, train_years: int | None = 4) -> gym.Env:
    panel_clean    = load_parquet(CLEAN_PANEL)
    panel_features = load_parquet(FEATURES_NORM)
    riskfree       = load_parquet(FEATURES_NORM)

    dates  = panel_clean.index.get_level_values(0).unique().sort_values()
    assets = get_assets_flat(get_asset_groups())
    rf_factor = riskfree["rf_daily_factor_raw"].reindex(dates).to_numpy()
    rf_rate   = riskfree["risk_free_rate_raw"].reindex(dates).to_numpy()

    spec = load_spec(spec_features)

    # Fenster: letzte N Jahre
    start_idx = 0
    end_idx_exclusive = len(dates)
    if train_years and train_years > 0:
        end_date = dates[-1]
        start_date = pd.Timestamp(end_date) - pd.DateOffset(years=int(train_years))
        start_idx = int(dates.searchsorted(start_date, side="left"))

    env = TradingEnv(
        panel_clean=panel_clean,
        panel_features=panel_features,
        dates=dates,
        assets=assets,
        spec=spec,
        state_builder=dict(build_state_for_date=build_state_for_date),
        portfolio=PortfolioLite(assets=assets, initial_cash=initial_cash),
        initial_cash=initial_cash,
        rf_factor=rf_factor,
        rf_rate=rf_rate,
        reward_kind="log",
        recorder=None,                 # <- kein Accounting
        start_idx=start_idx,
        end_idx_exclusive=end_idx_exclusive,
        validate_actions=True,
    )
    env = ActionMappingWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

# --- Modell-Fabrik ------------------------------------------------------------
def make_model(algo: str, env, seed: int = 42):
    policy = "MultiInputPolicy"
    if algo.lower() == "ppo":
        return PPO(
            policy, env,
            policy_kwargs=dict(features_extractor_class=CNN1DExtractor,
                               net_arch={"pi":[256,128], "vf":[256,128]}),
            n_steps=4096, batch_size=128, n_epochs=5, gae_lambda=0.95,
            gamma=0.999, learning_rate=3e-4, ent_coef=0.0, max_grad_norm=0.5,
            clip_range=0.2, verbose=0, seed=seed, device="auto",
        )
    elif algo.lower() == "sac":
        return SAC(
            policy, env,
            policy_kwargs=dict(features_extractor_class=CNN1DExtractor,
                               net_arch={"pi":[256,128], "qf":[256,128]}),
            learning_rate=3e-4, buffer_size=500_000, batch_size=256,
            gamma=0.999, tau=0.005, train_freq=128, gradient_steps=16,
            learning_starts=1_000, ent_coef="auto", verbose=0, seed=seed, device="auto",
        )
    else:
        raise ValueError("algo muss 'ppo' oder 'sac' sein")

# --- Hauptablauf --------------------------------------------------------------
def main(
    algo: str = "ppo",
    seed: int = 7,
    train_steps: int = 200_000,
    stop_after: int = 10_000,
    test_steps: int = 5_000,
    train_years: int = 4,
):
    set_seed(seed)

    # TRAINING (Zeit messen)
    env = DummyVecEnv([lambda: make_env(seed=seed, train_years=train_years)])
    model = make_model(algo, env, seed=seed)

    cb_list = []
    if stop_after and stop_after > 0:
        cb_list.append(StopAfterNSteps(max_timesteps=stop_after, verbose=1))
    callback = CallbackList(cb_list) if cb_list else None

    t0 = time.perf_counter()
    model.learn(total_timesteps=int(train_steps), callback=callback)
    t1 = time.perf_counter()
    print(f"[TIME] Training: {t1 - t0:.2f} s (requested={train_steps:,} steps, stop_after={stop_after:,})")

    # TEST (deterministische Inferenz; Zeit messen)
    test_env = make_env(seed=seed+123, train_years=train_years)  # einzelne Env ohne Vec
    obs, _ = test_env.reset(seed=seed+123)
    steps = 0
    t2 = time.perf_counter()
    done = False
    while steps < int(test_steps) and not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    t3 = time.perf_counter()
    print(f"[TIME] Test:     {t3 - t2:.2f} s (executed={steps:,} steps, done={done})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train_steps", type=int, default=200_000)
    ap.add_argument("--stop_after", type=int, default=10_000, help="Früher Abbruch fürs Timing (Steps).")
    ap.add_argument("--test_steps", type=int, default=5_000)
    ap.add_argument("--train_years", type=int, default=4)
    args = ap.parse_args()
    main(**vars(args))
