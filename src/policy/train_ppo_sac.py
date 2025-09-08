# -*- coding: utf-8 -*-
"""
Training-Script für PPO/SAC mit deiner TradingEnv:
- nutzt Dict-Observationen (MultiInputPolicy)
- mapped Agent-Aktionen via Softmax -> Gewichte (ActionMappingWrapper)
- loggt Accounting-Parquets + eine kleine Summary
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import gymnasium as gym
import os, torch, sys

# Stable-Baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from src.rl.cnn_extractor import CNN1DExtractor


from src.utils.paths import (
    CLEAN_PANEL, FEATURES_NORM, RISKFREE_NORM_FILE, SPEC_S0_YAML as spec_features , ACCOUNT_DIR,
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

# --- Repro: globale Flags/Threads ----------------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

# -----------------------------------------------------------------------------

def make_env(seed: int = 42, initial_cash: float = 1_000_000.0) -> gym.Env:
    panel_clean = load_parquet(CLEAN_PANEL)
    panel_features = load_parquet(FEATURES_NORM)
    riskfree    = load_parquet(RISKFREE_NORM_FILE)

    dates  = panel_clean.index.get_level_values(0).unique().sort_values()
    assets = get_assets_flat(get_asset_groups())
    A = len(assets)

    rf_factor = riskfree["daily_factor_360"].reindex(dates).to_numpy()
    rf_rate = riskfree["risk_free_annual_z"].reindex(dates).to_numpy()
    spec = load_spec(spec_features)
    state_builder = SimpleNamespace(build_state_for_date=build_state_for_date)

    portfolio = PortfolioLite(assets=assets, initial_cash=initial_cash)
    recorder  = AccountingRecorder(out_dir=Path(ACCOUNT_DIR))

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
        validate_actions=True,   # clip/norm als fallback
    )
    # Wichtig: Wrapper, damit der Agent Logits ausgeben darf
    env = ActionMappingWrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def train(algo: str = "ppo",
          total_timesteps: int = 200_000,
          seed: int = 42,
          tensorboard_log: str | None = None, use_eval: bool = False):
    """
    algo: 'ppo' oder 'sac'
    """
    # Reproduzierbarkeit & Run-Stamp
    set_seed(seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(ACCOUNT_DIR) / f"runs/{algo}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("TRAIN", to_file=str(run_dir / "train.log"))
    logger.info(
        f"seed={seed} algo={algo} steps={total_timesteps} tb={tensorboard_log} "
        f"python={sys.version.split()[0]} torch={torch.__version__} sb3={(PPO.__module__).split('.')[0]}")


    env = DummyVecEnv([lambda: make_env(seed=seed)])
    eval_env = DummyVecEnv([lambda: make_env(seed=seed + 123)]) if use_eval else None

    policy = "MultiInputPolicy"  # Dict-Observations: X, g_scalars, g_weights, position
    policy_kwargs = dict(
        features_extractor_class=CNN1DExtractor,
        # optional: features_extractor_kwargs=dict(channels=X, kernel_size=Y, ...),
    )

    if algo.lower() == "ppo":
        # Solide Startwerte; ggf. anpassen
        model = PPO(
            policy,
            env,
            policy_kwargs=policy_kwargs,
            n_steps=512,
            batch_size=128,
            n_epochs=4,
            gae_lambda=0.95,
            gamma=0.999,           # eher hoch bei Finanzdaten
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

    write_run_manifest(
        run_dir=run_dir,
        algo=algo,
        model=model,
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        deep=False,  # Server/Final: True setzen
    )

    # Eval-Callback (bewahrt bestes Modell)
    if use_eval:
        best_path = Path(ACCOUNT_DIR) / f"models/{algo}_{ts}"
        best_path.mkdir(parents=True, exist_ok=True)
        eval_cb = EvalCallback(eval_env, best_model_save_path=str(best_path),
                               log_path=str(best_path), eval_freq=10_000,
                               deterministic=True, render=False, n_eval_episodes=1)
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)
        model.save(best_path / "final_model")
    else:
        model.learn(total_timesteps=total_timesteps)

    # Kurze Summary aus den Parquets (vom letzten Run in env.recorder)
    # Hinweis: Recorder schreibt in ACCOUNT_DIR (global). Hier nur eine Mini-Nav-Zusammenfassung:

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
    peak = nav.cummax(); mdd = float(((peak - nav) / peak.replace(0, np.nan)).max())

    out_csv = Path(ACCOUNT_DIR) / f"summary_{algo}_{ts}.csv"
    pd.DataFrame([{
        "rounds": int(len(df)),
        "nav_start": nav0, "nav_end": navT, "total_return": ret_tot,
        "ann_mean_r_log": ann_mu, "ann_std_r_log": ann_sig, "max_drawdown": mdd
    }]).to_csv(out_csv, index=False)
    print(f"[DONE] Algo={algo}  Summary -> {out_csv}")

if __name__ == "__main__":
    # Beispiel: PPO 1e5 Schritte
    train(algo="ppo", total_timesteps=2_000, seed=7, tensorboard_log=None)
    # oder: train(algo="sac", total_timesteps=200_000)
    # str(Path(ACCOUNT_DIR) / "tb")
