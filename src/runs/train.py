# src/runs/train.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime
import shutil, json
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

import src.state.state_builder as sb
from src.env.data_builder import load_data_for_windows, build_env_segment
from src.rl.agent_policy import make_agent
from src.accounting.evaluator import compute_rewards_from_snapshots, RewardSpec, OnlineEvaluator
from src.utils import paths

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

# Helper für verschiedene Envs für HP Optimierung
def _make_seeded_env(panel, seg, spec, reward, seed_i):
    env = build_env_segment(panel, seg, state_spec=spec, reward_kind=reward,
                            with_recorder=False, out_dir=None)
    # Gymnasium: Reset mit Seed; Fallback auf .seed()
    try:
        env.reset(seed=seed_i)
    except TypeError:
        env.seed(seed_i)
    return env

def _vec_for_fold(fold, panel, spec, reward, n_envs, mode, seed_base):
    train_segs = list(fold["train"])
    if mode == "seeds":
        base_seg = train_segs[0]
        fns = [lambda s=seed_base+i: _make_seeded_env(panel, base_seg, spec, reward, s)
               for i in range(n_envs)]
    else:  # "segments": Segmente ggf. zyklisch auffüllen bis n_envs erreicht
        if n_envs <= len(train_segs):
            use_segs = train_segs[:n_envs]
        else:
            reps = (n_envs + len(train_segs) - 1) // len(train_segs)
            use_segs = (train_segs * reps)[:n_envs]
        fns = [lambda seg=use_segs[i], s=seed_base+i:
               _make_seeded_env(panel, seg, spec, reward, s) for i in range(n_envs)]
    Vec = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return Vec(fns)

def _load_trials(path):
    import yaml, json
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


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
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--env_mode", choices=["segments", "seeds"], default="segments")
    p.add_argument("--hpo_mode", choices=["none", "grid"], default="none")
    p.add_argument("--hpo_param_file", type=str, default=None)  # YAML/JSON: Liste von Trials
    p.add_argument("--hpo_splits", type=str, default=None)  # z.B. "0,5,9" (nur diese 3 Folds)

    args = p.parse_args()

    windows = list(_read_windows(args.strategy, args.splits))
    print("[DEBUG] _read_windows returned:", type(windows))
    try:
        print("[DEBUG] number of windows (len):", len(windows))
    except TypeError:
        print("[DEBUG] windows is not sized (generator). Convert to list to iterate safely.")
    print(f"[RUN] {len(windows)} splits loaded from {args.splits}")
    panel = load_data_for_windows(windows, strategy=args.strategy, features_source=args.features_source)
    spec = sb.load_spec(str(args.state_spec))

    # Beispiel: nur diese Folds in Stage A verwenden
    if args.hpo_splits:
        keep = [int(x) for x in args.hpo_splits.split(",")]
        windows = [windows[i] for i in keep]

    if args.strategy.lower() == "walkforward":
        print(f"[DATA] walk-forward -> using features_source={args.features_source}")
    else:
        # best effort: extract years (same logic as in data_builder)
        years = sorted({int(s[0][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])} |
                       {int(s[1][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])})
        print(f"[DATA] cpcv -> loading per-year parquet for years={years}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{args.algo}_{args.reward}_{Path(args.state_spec).stem}_{args.strategy}_{ts}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)
    manifest = {
        "algo": args.algo, "reward": args.reward, "state_spec": args.state_spec,
        "strategy": args.strategy, "splits": args.splits, "seed": args.seed,
        "timesteps": args.total_timesteps, "timestamp": ts
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    meta = {
        "cmdline_args": vars(args),  # Python argparse -> dict
        "splits_file": str(args.splits),
        "state_spec": str(args.state_spec),
        "algo": args.algo,
        "reward": args.reward,
    }
    # write JSON
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    # copy used splits yaml for audit
    try:
        shutil.copyfile(args.splits, run_dir / Path(args.splits).name)
    except Exception:
        pass

    # ---- HPO: Trials bestimmen ----
    trials = [{"_name": "single"}]  # Default: normaler Run ohne HPO
    if args.hpo_mode == "grid":
        assert args.hpo_param_file, "--hpo_param_file ist nötig bei hpo_mode=grid"
        trials = _load_trials(args.hpo_param_file)  # Liste von Dicts


    # ---------- WF: Modell einmal vor der Schleife bauen ----------
    model = None
    if args.strategy == "walkforward":
        first_seg = windows[0]["train"][0]
        init_env = DummyVecEnv([lambda: build_env_segment(
            panel, first_seg, state_spec=spec, reward_kind=args.reward,
            with_recorder=False, out_dir=None)])
        model = make_agent(args.algo, init_env, tensorboard_log=str(run_dir / "tb"), seed=args.seed)

    # TRAIN/TEST pro Fold
    for t_id, t in enumerate(trials):
        trial_tag = t.get("_name", f"trial{t_id:03d}")
        trial_dir = run_dir / trial_tag
        (trial_dir / "tb").mkdir(parents=True, exist_ok=True)

        # Basiskonfig für PPO holen (so wie ihr sie bisher an make_agent übergebt)
        ppo_kwargs =  {k: v for k, v in t.items() if not k.startswith("_")}
        # Trial-Overrides einmischen (Keys, die mit "_" beginnen, ignorieren)
        for k, v in t.items():
            if not k.startswith("_"):
                ppo_kwargs[k] = v

        for k, fold in enumerate(windows, 1):
            fold_dir = trial_dir / f"fold_{k:02d}"
            (fold_dir / "train").mkdir(parents=True, exist_ok=True)
            (fold_dir / "test").mkdir(parents=True, exist_ok=True)
            (fold_dir / "fold_meta.json").write_text(json.dumps(fold, indent=2))

            # === TRAIN: mehrere Episoden ohne Recorder ===
            train_vec = _vec_for_fold(fold, panel, spec, args.reward,
                                      n_envs=args.n_envs, mode=args.env_mode, seed_base=args.seed + 1000 * t_id)

            if args.strategy == "cpcv":
                model = make_agent(args.algo, train_vec, tensorboard_log=str(fold_dir / "tb"), seed=args.seed)
            else:
                model.set_env(train_vec)

            model.learn(total_timesteps=args.total_timesteps, progress_bar=False)

            try:
                rollout_size = model.n_steps * train_vec.num_envs
                assert rollout_size % model.batch_size == 0, \
                    f"rollout_size={rollout_size} nicht teilbar durch batch_size={model.batch_size}"
            except Exception:
                pass  # falls Algo ohne diese Attribute (SAC ok, PPO hat sie normalerweise)

            train_vec.close()


            # ===== TEST: from_snapshots =====
            if args.eval_mode == "from_snapshots":
                for i, seg in enumerate(fold["test"], 1):
                    y = str(seg[0])[:4]
                    print(y)
                    seg_dir = fold_dir / "test" / f"test_{y}"
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

                    #Make path relative to project/data (compute_rewards expects DATA_DIR / accounting_dir)
                    try:
                        acc_dir_rel = seg_dir.relative_to(paths.DATA_DIR)
                        print(f"First {acc_dir_rel}")
                    except Exception:
                        parts = seg_dir.parts
                        if parts and parts[0].lower() == "data":
                            acc_dir_rel = Path(*parts[1:])
                            print(f"Second {acc_dir_rel}")
                        else:
                            try:
                                acc_dir_rel = seg_dir.relative_to(Path.cwd() / "data")
                                print(f"Thrid {acc_dir_rel}")
                            except Exception:
                                raise RuntimeError(
                                    f"Kann {seg_dir} nicht in einen Pfad relativ zu project/data umwandeln; prüfe run_root.")

                    compute_rewards_from_snapshots(
                        accounting_dir=acc_dir_rel,
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
