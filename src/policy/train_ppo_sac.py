# -*- coding: utf-8 -*-
"""
Training-Script für PPO/SAC mit deiner TradingEnv:
- Dict-Observationen (MultiInputPolicy) + CNN1DExtractor
- ActionMappingWrapper: Agent-Logits -> Portfolio-Gewichte
- Accounting-Parquets + kurze Summary
- Profiling: Steps/Sek + frühzeitiger Stopp nach N Steps (StopAfterNSteps)
"""

from __future__ import annotations

# --- Standard / Basics -------------------------------------------------------
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import os, sys, time
import argparse

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

# --- Stable-Baselines3 -------------------------------------------------------
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

# --- Projekt-Imports ---------------------------------------------------------
from src.rl.cnn_extractor import CNN1DExtractor
from src.utils.paths import (
    CLEAN_PANEL, FEATURES_NORM, RISKFREE_NORM_FILE,
    SPEC_S0_YAML as spec_features, ACCOUNT_DIR,
    get_asset_groups, get_assets_flat,
)
from src.utils.parquet_io import load_parquet
from src.state.state_builder import load_spec, build_state_for_date
from src.env.trading_env import TradingEnv
from src.accounting.recorder import AccountingRecorder
from src.portfolio.broker import PortfolioLite
from src.rl.wrapper import ActionMappingWrapper
from src.utils.helpers import set_seed, get_logger, write_run_manifest
from src.accounting.evaluator import compute_rewards_from_snapshots
from src.accounting.reward import RewardSpec


# === Profiling-Callbacks =====================================================

class SpeedLogger(BaseCallback):
    """Loggt Steps/Sekunde und Sekunden pro 1e4 Steps."""
    def __init__(self, window: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.window = int(window)
        self._last_t = None
        self._last_n = 0

    def _on_training_start(self) -> None:
        self._last_t = time.time()
        self._last_n = 0

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n - self._last_n >= self.window:
            now = time.time()
            dt = max(now - self._last_t, 1e-9)
            delta = n - self._last_n
            sps = delta / dt
            wall_per_1e4 = 10_000.0 / sps
            if self.verbose:
                print(f"[PROFILE] {n:,} steps | {sps:,.1f} steps/s | {wall_per_1e4:,.2f} s pro 1e4 steps")
            self._last_t = now
            self._last_n = n
        return True


class StopAfterNSteps(BaseCallback):
    """Bricht das Training ab, sobald die gegebene Step-Schwelle erreicht ist."""
    def __init__(self, max_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_timesteps = int(max_timesteps)

    def _on_step(self) -> bool:
        n = int(self.model.num_timesteps)
        if n >= self.max_timesteps:
            if self.verbose:
                print(f"[PROFILE] Stop nach {n:,} Steps (Schwelle={self.max_timesteps:,})")
            return False  # Training beenden
        return True


# === Repro/Threads (konservativ; bei Bedarf anpassen) ========================
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# === Env-Fabrik ==============================================================
def make_env(seed: int = 42, initial_cash: float = 1_000_000.0, train_years: int | None = None) -> gym.Env:
    """Baut die Trading-Umgebung inkl. Wrapper & Monitor."""
    # Daten laden
    panel_clean    = load_parquet(CLEAN_PANEL)
    panel_features = load_parquet(FEATURES_NORM)
    riskfree       = load_parquet(RISKFREE_NORM_FILE)

    dates  = panel_clean.index.get_level_values(0).unique().sort_values()
    assets = get_assets_flat(get_asset_groups())
    rf_factor = riskfree["daily_factor_360"].reindex(dates).to_numpy()
    rf_rate   = riskfree["risk_free_annual_z"].reindex(dates).to_numpy()

    # State-Spec
    spec = load_spec(spec_features)
    state_builder = SimpleNamespace(build_state_for_date=build_state_for_date)

    # Portfolio + Accounting
    portfolio = PortfolioLite(assets=assets, initial_cash=initial_cash)
    recorder  = AccountingRecorder(out_dir=Path(ACCOUNT_DIR))

    # --- Fenster wählen ---
    start_idx = 0
    end_idx_exclusive = len(dates)
    if train_years and train_years > 0:
        end_date = dates[-1]
        start_date = pd.Timestamp(end_date) - pd.DateOffset(years=int(train_years))
        start_idx = int(dates.searchsorted(start_date, side="left"))
        # Ende bleibt letzter Tag:
        end_idx_exclusive = len(dates)

    # Env
    env = TradingEnv(
        panel_clean=panel_clean,
        panel_features=panel_features,
        dates=dates,
        assets=assets,
        spec=spec,
        state_builder=state_builder,
        portfolio=portfolio,
        initial_cash=initial_cash,
        rf_factor=rf_factor,
        rf_rate=rf_rate,
        reward_kind="log",
        recorder=recorder,
        start_idx=0,
        end_idx_exclusive=len(dates),
        validate_actions=True,  # clip/norm fallback im Broker
    )

    # Wrapper: Agent-Logits -> Gewichte; Monitoring/Seeding
    env = ActionMappingWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# === Training ================================================================
def train(algo: str = "ppo",
          total_timesteps: int = 200_000,
          seed: int = 42,
          tensorboard_log: str | None = None,
          use_eval: bool = False,
          max_steps_profile: int = 0):
    """
    Startet Training für PPO oder SAC.
    - total_timesteps: gesamte Env-Steps (über viele Episoden/Rollouts)
    - max_steps_profile > 0: Profiling-Stop nach N Steps (frühes Ende)
    """
    # Seed + Run-Verzeichnis + Logger
    set_seed(seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(ACCOUNT_DIR) / f"runs/{algo}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("TRAIN", to_file=str(run_dir / "train.log"))
    logger.info(
        f"seed={seed} algo={algo} steps={total_timesteps} tb={tensorboard_log} "
        f"python={sys.version.split()[0]} torch={torch.__version__} sb3={(PPO.__module__).split('.')[0]}"
    )

    # VecEnvs
    env = DummyVecEnv([lambda: make_env(seed=seed, train_years=4)])
    eval_env = DummyVecEnv([lambda: make_env(seed=seed + 123, train_years=4)]) if use_eval else None

    # Policy (Dict-Obs)
    policy = "MultiInputPolicy"
    policy_kwargs = dict(
        features_extractor_class=CNN1DExtractor,
        net_arch={"pi": [256, 128], "vf": [256, 128]},
    )

    # Modell
    if algo.lower() == "ppo":
        model = PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.999,
            learning_rate=3e-4,
            ent_coef=0.0,
            max_grad_norm=0.5,
            clip_range=0.2,
            verbose=1,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device="auto",
        )
    elif algo.lower() == "sac":
        model = SAC(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            gamma=0.999,
            tau=0.005,
            train_freq=64,
            gradient_steps=64,
            learning_starts=5_000,
            ent_coef="auto",
            verbose=1,
            seed=seed,
            tensorboard_log=tensorboard_log,
            device="auto",
        )
    else:
        raise ValueError("algo muss 'ppo' oder 'sac' sein")

    # Manifest (Run-Metadaten)
    write_run_manifest(
        run_dir=run_dir,
        algo=algo,
        model=model,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        deep=False,  # für Final-Runs ggf. True
    )

    # Callbacks: Speed + optional Profiling-Stop + optional Eval
    cbs = [SpeedLogger(window=10_000, verbose=1)]
    cbs: list[BaseCallback] = [SpeedLogger(window=10_000, verbose=1)]
    if max_steps_profile and max_steps_profile > 0:
        cbs.append(StopAfterNSteps(max_timesteps=int(max_steps_profile), verbose=1))
    if use_eval:
        best_path = Path(ACCOUNT_DIR) / f"models/{algo}_{ts}"
        best_path.mkdir(parents=True, exist_ok=True)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(best_path),
            log_path=str(best_path),
            eval_freq=10_000,
            deterministic=True,
            render=False,
            n_eval_episodes=1,
        )
        cbs.append(eval_cb)
    callback = CallbackList(cbs)

    # Lernen (Profiling-Stop beendet ggf. früher)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Kurze Summary (aus Accounting-Snapshots)
    rewards = compute_rewards_from_snapshots(
        accounting_dir=Path(ACCOUNT_DIR),
        spec=RewardSpec(kind="log"),
        out_name=f"rewards_{algo}_{ts}.parquet",
    )
    df = rewards.sort_values("round").reset_index(drop=True)
    nav = df["nav_t"].astype(float).ffill()
    rlog = df["r_log_t"].astype(float)

    nav0 = float(nav.iloc[1]) if len(nav) > 1 else float(nav.iloc[0])
    navT = float(nav.iloc[-1])
    ret_tot = (navT / max(1e-12, nav0)) - 1.0
    ann_mu = float(rlog.mean(skipna=True)) * 252.0
    ann_sig = float(rlog.std(skipna=True)) * (252.0 ** 0.5)
    peak = nav.cummax()
    mdd = float(((peak - nav) / peak.replace(0, np.nan)).max())

    out_csv = Path(ACCOUNT_DIR) / f"summary_{algo}_{ts}.csv"
    pd.DataFrame([{
        "rounds": int(len(df)),
        "nav_start": nav0, "nav_end": navT, "total_return": ret_tot,
        "ann_mean_r_log": ann_mu, "ann_std_r_log": ann_sig, "max_drawdown": mdd
    }]).to_csv(out_csv, index=False)
    print(f"[DONE] Algo={algo}  Summary -> {out_csv}")


# === CLI =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tensorboard_log", type=str, default=None)
    ap.add_argument("--use_eval", action="store_true")
    ap.add_argument(
        "--max_steps_profile", type=int, default=0,
        help="Wenn >0: Training vorzeitig nach so vielen Env-Steps stoppen und Speed loggen."
    )
    args = ap.parse_args()

    train(
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        tensorboard_log=args.tensorboard_log,
        use_eval=args.use_eval,
        max_steps_profile=args.max_steps_profile,
    )
