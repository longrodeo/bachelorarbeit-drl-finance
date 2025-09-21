# src/runs/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime

from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv
import src.state.state_builder as sb
from src.env.data_builder import load_data_for_windows, build_env_segment
from src.rl.agent_policy import make_agent
from src.accounting.evaluator import compute_rewards_from_snapshots, RewardSpec, OnlineEvaluator

# helper: robustes Reset/Step-Handling für eval
def _unpack_reset(env):
    res = env.reset()
    return res if not isinstance(res, tuple) else res[0]

def _unpack_step(step_res):
    # step_res kann (obs, reward, done, info) oder
    # (obs, reward, terminated, truncated, info) sein
    if not isinstance(step_res, tuple):
        raise RuntimeError("Unexpected env.step() return type")
    if len(step_res) == 4:
        obs, reward, done, info = step_res
        return obs, reward, bool(done), info
    elif len(step_res) == 5:
        obs, reward, terminated, truncated, info = step_res
        return obs, reward, bool(terminated or truncated), info
    else:
        raise RuntimeError(f"Unexpected step tuple length: {len(step_res)}")

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
    p.add_argument("--total_timesteps", type=int, default=5_000)
    p.add_argument("--run_root", default="data/accounting/runs")
    p.add_argument("--eval_mode", choices=["from_snapshots", "online"], default="from_snapshots")
    p.add_argument("--features_source", default="features_v1_raw_z")
    args = p.parse_args()

    windows = _read_windows(args.strategy, args.splits)
    panel = load_data_for_windows(windows, strategy=args.strategy, features_source=args.features_source)
    spec = sb.load_spec(str(args.state_spec))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{args.algo}_{args.reward}_{Path(args.state_spec).stem}_{args.strategy}_{ts}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    manifest = {
        "algo": args.algo, "reward": args.reward, "state_spec": args.state_spec,
        "strategy": args.strategy, "splits": args.splits, "seed": args.seed,
        "timesteps": args.total_timesteps, "timestamp": ts
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---------- WF: Modell einmal vor der Schleife bauen ----------
    model = None
    if args.strategy == "walkforward":
        first_seg = windows[0]["train"][0]
        init_env = DummyVecEnv([lambda: build_env_segment(
            panel, first_seg, state_spec=spec, reward_kind=args.reward,
            with_recorder=False, out_dir=None)])
        model = make_agent(args.algo, init_env, tensorboard_log=str(run_dir / "tb"), seed=args.seed)

    # TRAIN/TEST pro Fold
    for k, fold in enumerate(windows, 1):
        fold_dir = run_dir / f"fold_{k:02d}"
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "test").mkdir(parents=True, exist_ok=True)

        # === TRAIN: mehrere Episoden ohne Recorder ===
        def _make(seg):
            return build_env_segment(panel, seg, state_spec=spec, reward_kind=args.reward,
                                     with_recorder=False, out_dir=None)
        train_vec = DummyVecEnv([partial(_make, seg) for seg in fold["train"]])

        if args.strategy == "cpcv":
            # --- CPCV: pro Fold NEUES Modell, direkt mit train_vec bauen
            model = make_agent(args.algo, train_vec, tensorboard_log=str(fold_dir / "tb"), seed=args.seed)
        else:
            # --- WF: bestehendes Modell weiterverwenden
            # num_envs muss übereinstimmen; bei WF ist das i. d. R. der Fall.
            model.set_env(train_vec)

        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        train_vec.close()

        # ===== TEST: from_snapshots =====
        if args.eval_mode == "from_snapshots":
            for i, seg in enumerate(fold["test"], 1):
                seg_dir = fold_dir / "test" / f"seg_{i:02d}"
                env = build_env_segment(panel, seg, state_spec=spec, reward_kind=args.reward,
                                        with_recorder=True, out_dir=seg_dir)

                # Gymnasium: reset() may return (obs, info)
                obs = _unpack_reset(env)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = _unpack_step(env.step(action))
                # close afterwards
                env.close()

            compute_rewards_from_snapshots(
                accounting_dir=(fold_dir / "test"),
                spec=RewardSpec(kind=args.reward),
                out_name="rewards.parquet"
            )

        # ===== TEST: online =====
        else:  # online
            online = OnlineEvaluator(kind=args.reward)
            for i, seg in enumerate(fold["test"], 1):
                env = build_env_segment(panel, seg, state_spec=spec, reward_kind=args.reward,
                                        with_recorder=False, out_dir=None)

                obs = _unpack_reset(env)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = _unpack_step(env.step(action))

                    nav_t = info.get("value") if isinstance(info, dict) else None
                    r_log = info.get("r_log") if isinstance(info, dict) else None
                    online.update(nav_t=nav_t, r_log_t=r_log)
                env.close()

            df = online.finalize()
            (fold_dir / "test").mkdir(exist_ok=True, parents=True)
            df[["r_log_t", "icvar_t", "delta_mdd_t", "reward_t"]].describe().to_csv(
                fold_dir / "test" / "summary_test.csv")

if __name__ == "__main__":
    main()
