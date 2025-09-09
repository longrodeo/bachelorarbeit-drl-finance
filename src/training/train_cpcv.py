# train\_cpcv.py (Baseline)


import os, argparse, yaml, math
from datetime import datetime, timedelta
import numpy as np
from stable_baselines3 import PPO, SAC
from src.env.trading_env import TradingEnv
from src.rl.cnn_extractor import CNN1DExtractor

# --- Helpers (minimal) ---

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    return ap.parse_args()


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dates_from_panel(panel_path, date_col="date"):
    # Erwartet: Parquet mit MultiIndex [date, asset] ODER eine Spalte 'date'.
    # Wir lesen nur die Datumsachse über die Env (billig), um Abhängigkeiten gering zu halten.
    env = TradingEnv(data_path=os.path.dirname(panel_path), t_plus_one=True)
    idx = env.get_date_index() if hasattr(env, "get_date_index") else None
    if idx is None:
        raise RuntimeError("TradingEnv muss get_date_index() bereitstellen oder passe die Datumsquelle hier an.")
    return np.array(idx)


def make_purged_wf_splits(dates, n_folds=5, embargo_days=5):
    # CPCV-lite: Walk-Forward mit Purge (Train = bis vor Test-Start-Embargo)
    n = len(dates)
    fold_sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]
    starts = np.cumsum([0] + fold_sizes[:-1])
    splits = []
    for k, start in enumerate(starts):
        end = start + fold_sizes[k]
        test_start = start
        test_end = end
        if test_start == 0:
            # kein Training vor dem ersten Testfenster
            continue
        train_end_idx = max(0, test_start - embargo_days)
        train_start_date = dates[0]
        train_end_date = dates[train_end_idx - 1] if train_end_idx > 0 else dates[0]
        test_start_date = dates[test_start]
        test_end_date = dates[test_end - 1]
        splits.append(dict(fold=k,
                           train_from=str(train_start_date), train_to=str(train_end_date),
                           test_from=str(test_start_date), test_to=str(test_end_date)))
    return splits


def build_model(algo, env, policy_kwargs, hp):
    if algo.lower() == "ppo":
        return PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **hp)
    elif algo.lower() == "sac":
        return SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, **hp)
    else:
        raise ValueError(f"Unbekannter Algo: {algo}")


def simple_rollout(model, env, deterministic=True, max_steps=2_000_000):
    obs, _ = env.reset()
    rewards = []
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        if done or truncated:
            break
    return rewards


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config)

    out_root = cfg.get("output_dir", "outputs/cpcv")
    os.makedirs(out_root, exist_ok=True)

    # Datumsachse bestimmen (über Env oder direkt über Panel)
    # Vereinfachung: wir lassen die Env die Daten laden; falls du lieber das Panel nutzt, tausche get_dates_from_panel aus.
    panel_path = cfg["data"].get("panel_path", "data/raw/panel.parquet")
    embargo_days = int(cfg["cv"].get("embargo_days", 5))
    n_folds = int(cfg["cv"].get("n_folds", 5))

    # Hole Datumsindex aus der Env (du kannst optional date_start/date_end aus cfg begrenzen)
    env_for_index = TradingEnv(data_path=cfg["data"]["data_path"], t_plus_one=True)
    dates = np.array(env_for_index.get_date_index()) if hasattr(env_for_index, "get_date_index") else None
    if dates is None:
        raise RuntimeError("TradingEnv.get_date_index() benötigt. Alternativ: eigenständig Panel lesen und Datumsindex extrahieren.")

    # Optional beschneiden
    d0 = cfg["data"].get("date_start")
    d1 = cfg["data"].get("date_end")
    if d0:
        dates = dates[dates >= d0]
    if d1:
        dates = dates[dates <= d1]

    splits = make_purged_wf_splits(dates, n_folds=n_folds, embargo_days=embargo_days)

    # Gemeinsame Policy-Settings (CNN1D)
    policy_kwargs = dict(features_extractor_class=CNN1DExtractor)

    for run in cfg["runs"]:
        algo = run["algo"]  # "ppo" | "sac"
        hp = run.get("hyperparams", {})
        total_timesteps = int(run.get("total_timesteps", 200_000))
        seeds = run.get("seeds", [42])

        for seed in seeds:
            for sp in splits:
                tag = f"{algo}_fold{sp['fold']}_seed{seed}"
                out_dir = os.path.join(out_root, tag)
                os.makedirs(out_dir, exist_ok=True)

                # Train-Env
                train_env = TradingEnv(
                    data_path=cfg["data"]["data_path"],
                    cost_bps=cfg["env"].get("cost_bps", 25),
                    t_plus_one=cfg["env"].get("t_plus_one", True),
                    date_from=sp["train_from"],
                    date_to=sp["train_to"],
                    seed=seed,
                )

                model = build_model(algo, train_env, policy_kwargs, dict(seed=seed, verbose=1, **hp))
                model.learn(total_timesteps=total_timesteps)
                model.save(os.path.join(out_dir, "model"))

                # Test-Env & einfache Rollout-Eval (optional)
                test_env = TradingEnv(
                    data_path=cfg["data"]["data_path"],
                    cost_bps=cfg["env"].get("cost_bps", 25),
                    t_plus_one=cfg["env"].get("t_plus_one", True),
                    date_from=sp["test_from"],
                    date_to=sp["test_to"],
                    seed=seed,
                    eval_mode=True,
                )
                rewards = simple_rollout(model, test_env, deterministic=True)
                with open(os.path.join(out_dir, "rewards.txt"), "w") as f:
                    for r in rewards:
                        f.write(f"{r}\n")

                # Marker
                with open(os.path.join(out_dir, "meta.txt"), "w") as f:
                    f.write(f"algo={algo}\nseed={seed}\nfold={sp['fold']}\ntrain={sp['train_from']}..{sp['train_to']}\n"
                            f"test={sp['test_from']}..{sp['test_to']}\nsteps={total_timesteps}\n")

    print("Fertig: CPCV (purged WF) Training abgeschlossen.")
