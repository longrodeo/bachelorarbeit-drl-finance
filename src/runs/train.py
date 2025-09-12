# src/runs/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.data_builder import load_data_for_windows, build_env_segment
from src.rl.agent_policy import make_agent
from src.accounting.evaluator import compute_rewards_from_snapshots, RewardSpec, OnlineEvaluator

def _read_windows(strategy: str, path: str):
    if strategy == "cpcv":
        from src.splits.cpcv import iter_windows_from_yaml
    elif strategy == "walkforward":
        from src.splits.walkforward import iter_windows_from_yaml
    else:
        raise ValueError(f"strategy not supported: {strategy}")
    return iter_windows_from_yaml(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["cpcv", "walkforward"], required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--state_spec", required=True)                  # z.B. config/state_config/state0.yml
    p.add_argument("--reward", choices=["log", "icvar", "icvar_dd"], default="log")
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--run_root", default="data/accounting/runs")
    p.add_argument("--eval_mode", choices=["from_snapshots", "online"], default="from_snapshots")
    args = p.parse_args()

    windows = _read_windows(args.strategy, args.splits)
    panel = load_data_for_windows(windows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{args.algo}_{args.reward}_{Path(args.state_spec).stem}_{args.strategy}_{ts}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    manifest = {
        "algo": args.algo, "reward": args.reward, "state_spec": args.state_spec,
        "strategy": args.strategy, "splits": args.splits, "seed": args.seed,
        "timesteps": args.total_timesteps, "timestamp": ts
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Ein "neutraler" Env für die Initialisierung des Modells.
    # Wir nehmen das erste Train-Segment des ersten Folds.
    first_seg = windows[0]["train"][0]
    init_env = DummyVecEnv([lambda: build_env_segment(panel, first_seg,
                         state_spec=args.state_spec, reward_kind=args.reward,
                         with_recorder=False, out_dir=None)])
    model = make_agent(args.algo, init_env, tensorboard_log=str(run_dir / "tb"), seed=args.seed)

    # TRAIN/TEST pro Fold
    for k, fold in enumerate(windows, 1):
        fold_dir = run_dir / f"fold_{k:02d}"
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "test").mkdir(parents=True, exist_ok=True)

        # === TRAIN: mehrere Episoden ohne Recorder ===
        train_envs = [lambda seg=seg: build_env_segment(
            panel, seg, state_spec=args.state_spec, reward_kind=args.reward,
            with_recorder=False, out_dir=None) for seg in fold["train"]]
        train_vec = DummyVecEnv(train_envs)
        model.set_env(train_vec)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        train_vec.close()

        # === TEST: mehrere Episoden mit Recorder, nur Predict ===
        if args.eval_mode == "from_snapshots":
            for i, seg in enumerate(fold["test"], 1):
                seg_dir = fold_dir / "test" / f"seg_{i:02d}"
                env = build_env_segment(panel, seg, state_spec=args.state_spec, reward_kind=args.reward,
                                        with_recorder=True, out_dir=seg_dir)
                obs = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = env.step(action)
                env.close()

            compute_rewards_from_snapshots(
                accounting_dir=(fold_dir / "test"),
                spec=RewardSpec(kind=args.reward),
                out_name="rewards.parquet"
            )

        else:  # online
            online = OnlineEvaluator(kind=args.reward)
            for i, seg in enumerate(fold["test"], 1):
                env = build_env_segment(panel, seg, state_spec=args.state_spec, reward_kind=args.reward,
                                        with_recorder=False, out_dir=None)
                obs = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, info = env.step(action)
                    # Wichtig: hier deine Info-Keys verwenden (NAV / r_log)
                    nav_t = info.get("value")
                    r_log = info.get("r_log")
                    online.update(nav_t=nav_t, r_log_t=r_log)
                env.close()
            # Optional: Summary speichern
            df = online.finalize()
            (fold_dir / "test").mkdir(exist_ok=True, parents=True)
            df[["r_log_t", "icvar_t", "delta_mdd_t", "reward_t"]].describe().to_csv(
                fold_dir / "test" / "summary_test.csv")


if __name__ == "__main__":
    main()
