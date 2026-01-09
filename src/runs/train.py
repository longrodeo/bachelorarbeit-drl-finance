# src/runs/train.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import shutil, json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.utils.helpers import set_seed, get_logger
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
    p.add_argument("--wf_mode", choices=["refit", "warm"], default="refit")  # nur für walkforward
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
    set_seed(args.seed)

    # Run-Ordner und Logger vorbereiten
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{args.algo}_{args.reward}_{Path(args.state_spec).stem}_{args.strategy}_{ts}"
    (run_dir / "tb").mkdir(parents=True, exist_ok=True)

    logger = get_logger(name="TRAIN", to_file=run_dir / "train.log")
    logger.info("Run gestartet")
    logger.info("Run-Dir: %s", run_dir)
    logger.info("Args: %s", vars(args))

    try:
        # Manifest und Meta schreiben
        manifest = {
            "algo": args.algo,
            "reward": args.reward,
            "state_spec": args.state_spec,
            "strategy": args.strategy,
            "wf_mode": args.wf_mode,
            "splits": args.splits,
            "seed": args.seed,
            "timesteps": args.total_timesteps,
            "features_source": args.features_source,
            "n_envs": args.n_envs,
            "env_mode": args.env_mode,
            "eval_mode": args.eval_mode,
            "hpo_mode": args.hpo_mode,
            "hpo_param_file": args.hpo_param_file,
            "hpo_splits": args.hpo_splits,
            "timestamp": ts,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        meta = {
            "cmdline_args": vars(args),  # Python argparse -> dict
            "splits_file": str(args.splits),
            "state_spec": str(args.state_spec),
            "algo": args.algo,
            "reward": args.reward,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # copy used splits yaml for audit
        if args.hpo_mode == "grid" and args.hpo_param_file:
            try:
                shutil.copyfile(args.hpo_param_file, run_dir / Path(args.hpo_param_file).name)
            except Exception as e:
                logger.warning("Konnte hpo_param_file nicht kopieren: %s", e)

        try:
            shutil.copyfile(args.splits, run_dir / Path(args.splits).name)
        except Exception as e:
            logger.warning("Konnte splits-Datei nicht kopieren: %s", e)

        # Splits lesen
        windows = list(_read_windows(args.strategy, args.splits))
        logger.debug("[_read_windows] returned type: %s", type(windows))
        logger.info("[RUN] %d splits loaded from %s", len(windows), args.splits)

        # Beispiel: nur diese Folds in Stage A verwenden
        if args.hpo_splits:
            keep = [int(x) for x in args.hpo_splits.split(",")]
            windows = [windows[i] for i in keep]
            logger.info("HPO_splits aktiv: %s -> %d verwendete Folds", args.hpo_splits, len(windows))

        panel = load_data_for_windows(
            windows,
            strategy=args.strategy,
            features_source=args.features_source,
        )
        spec = sb.load_spec(str(args.state_spec))
        logger.info("Panel und state_spec geladen")



        if args.strategy.lower() == "walkforward":
            logger.info("[DATA] walk-forward -> using features_source=%s", args.features_source)
        else:
            # best effort: extract years (same logic as in data_builder)
            years = sorted(
                {int(s[0][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])}
                | {int(s[1][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])}
            )
            logger.info("[DATA] cpcv -> loading per-year parquet for years=%s", years)

        # ---- HPO: Trials bestimmen ----
        trials = [{"_name": "single"}]  # Default: normaler Run ohne HPO
        if args.hpo_mode == "grid":
            assert args.hpo_param_file, "--hpo_param_file ist nötig bei hpo_mode=grid"
            trials = _load_trials(args.hpo_param_file)  # Liste von Dicts
            logger.info("HPO-Modus: grid, Trials=%d, Param-File=%s", len(trials), args.hpo_param_file)
        else:
            logger.info("HPO-Modus: none (Single-Run)")

        # TRAIN/TEST pro Fold
        for t_id, t in enumerate(trials):
            trial_tag = t.get("_name", f"trial{t_id:03d}")
            trial_dir = run_dir / trial_tag
            (trial_dir / "tb").mkdir(parents=True, exist_ok=True)

            logger.info("Starte Trial %s (%d/%d)", trial_tag, t_id + 1, len(trials))
            model = None  # pro Trial neu (wichtig für grid + WF)

            try:
                # Basiskonfig für PPO/SAC holen
                ppo_kwargs = {k: v for k, v in t.items() if not k.startswith("_")}
                # Trial-Overrides einmischen (Keys, die mit "_" beginnen, ignorieren)

                # Trial-HPs persistent speichern (für Repro / BA-Doku)
                (trial_dir / "hparams.json").write_text(json.dumps(ppo_kwargs, indent=2), encoding="utf-8")
                logger.info("Trial %s hparams: %s", trial_tag, ppo_kwargs)

                for k, fold in enumerate(windows, 1):
                    fold_dir = trial_dir / f"fold_{k:02d}"
                    (fold_dir / "train").mkdir(parents=True, exist_ok=True)
                    (fold_dir / "test").mkdir(parents=True, exist_ok=True)
                    (fold_dir / "fold_meta.json").write_text(json.dumps(fold, indent=2))

                    logger.info(
                        "Starte Training für Trial %s, Fold %02d/%02d",
                        trial_tag,
                        k,
                        len(windows),
                    )

                    # === TRAIN: mehrere Episoden ohne Recorder ===
                    train_vec = _vec_for_fold(
                        fold,
                        panel,
                        spec,
                        args.reward,
                        n_envs=args.n_envs,
                        mode=args.env_mode,
                        seed_base=args.seed + 1000 * t_id,
                    )

                    if args.strategy == "cpcv" or (args.strategy == "walkforward" and args.wf_mode == "refit"):
                        model = make_agent(
                            args.algo,
                            train_vec,
                            tensorboard_log=str(fold_dir / "tb"),
                            seed=args.seed,
                            algo_kwargs=ppo_kwargs,
                        )
                    else:
                        # walkforward warm-start: Gewichte/Optimizer über Folds behalten
                        if model is None:
                            model = make_agent(
                                args.algo,
                                train_vec,
                                tensorboard_log=str(trial_dir / "tb"),
                                seed=args.seed,
                                algo_kwargs=ppo_kwargs,
                            )
                        else:
                            model.set_env(train_vec)

                    # Effective HPs aus dem echten SB3-Model dumpen (Proof gegen Defaults)
                    # learning_rate als Float (Schedule -> Wert)
                    lr_val = None
                    if hasattr(model, "lr_schedule") and callable(getattr(model, "lr_schedule")):
                        lr_val = float(model.lr_schedule(1.0))
                    elif isinstance(getattr(model, "learning_rate", None), (int, float)):
                        lr_val = float(model.learning_rate)

                    cr = getattr(model, "clip_range", None)
                    cr_val = float(cr(1.0)) if callable(cr) else (float(cr) if isinstance(cr, (int, float)) else cr)

                    eff = {
                        "algo": args.algo,
                        "trial": trial_tag,
                        "fold": k,
                        "learning_rate": lr_val,
                        "n_steps": getattr(model, "n_steps", None),
                        "batch_size": getattr(model, "batch_size", None),
                        "n_epochs": getattr(model, "n_epochs", None),
                        "gamma": getattr(model, "gamma", None),
                        "gae_lambda": getattr(model, "gae_lambda", None),
                        "ent_coef": getattr(model, "ent_coef", None),
                        "vf_coef": getattr(model, "vf_coef", None),
                        "max_grad_norm": getattr(model, "max_grad_norm", None),
                        "clip_range": cr_val,
                    }
                    (fold_dir / "effective_hparams.json").write_text(json.dumps(eff, indent=2), encoding="utf-8")

                    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, reset_num_timesteps=not (args.strategy == "walkforward" and args.wf_mode == "warm"))
                    logger.info(
                        "Training abgeschlossen: Trial %s, Fold %02d, timesteps=%d",
                        trial_tag,
                        k,
                        args.total_timesteps,
                    )

                    try:
                        rollout_size = model.n_steps * train_vec.num_envs
                        assert rollout_size % model.batch_size == 0, \
                            f"rollout_size={rollout_size} nicht teilbar durch batch_size={model.batch_size}"
                    except Exception:
                        # falls Algo ohne diese Attribute (SAC ok, PPO hat sie normalerweise)
                        logger.debug("rollout_size/batch_size-Check übersprungen (fehlende Attribute)")

                    train_vec.close()

                    # ===== TEST: from_snapshots =====
                    if args.eval_mode == "from_snapshots":
                        for i, seg in enumerate(fold["test"], 1):
                            y = str(seg[0])[:4]
                            seg_dir = fold_dir / "test" / f"test_{y}"
                            logger.info(
                                "Eval (from_snapshots): Trial %s, Fold %02d, Test-Segment %d, Jahr=%s",
                                trial_tag,
                                k,
                                i,
                                y,
                            )

                            env = build_env_segment(
                                panel,
                                seg,
                                state_spec=spec,
                                reward_kind=args.reward,
                                with_recorder=True,
                                out_dir=seg_dir,
                            )

                            # Gymnasium: reset() may return (obs, info)
                            obs = _unpack_reset(env)
                            done = False
                            while not done:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, reward, done, info = _unpack_step(env.step(action))
                            # close afterwards
                            env.close()

                            # Make path relative to project/data (compute_rewards expects DATA_DIR / accounting_dir)
                            try:
                                acc_dir_rel = seg_dir.relative_to(paths.DATA_DIR)
                                logger.debug("Relativer Pfad (First): %s", acc_dir_rel)
                            except Exception:
                                parts = seg_dir.parts
                                if parts and parts[0].lower() == "data":
                                    acc_dir_rel = Path(*parts[1:])
                                    logger.debug("Relativer Pfad (Second): %s", acc_dir_rel)
                                else:
                                    try:
                                        acc_dir_rel = seg_dir.relative_to(Path.cwd() / "data")
                                        logger.debug("Relativer Pfad (Third): %s", acc_dir_rel)
                                    except Exception:
                                        logger.exception(
                                            "Kann %s nicht relativ zu project/data umwandeln; prüfe run_root.",
                                            seg_dir,
                                        )
                                        raise RuntimeError(
                                            f"Kann {seg_dir} nicht in einen Pfad relativ zu project/data umwandeln; prüfe run_root."
                                        )

                            compute_rewards_from_snapshots(
                                accounting_dir=acc_dir_rel,
                                spec=RewardSpec(kind=args.reward),
                                out_name="rewards.parquet",
                            )

                    # ===== TEST: online =====
                    else:  # online
                        online = OnlineEvaluator(kind=args.reward)
                        for i, seg in enumerate(fold["test"], 1):
                            logger.info(
                                "Eval (online): Trial %s, Fold %02d, Test-Segment %d",
                                trial_tag,
                                k,
                                i,
                            )
                            env = build_env_segment(
                                panel,
                                seg,
                                state_spec=spec,
                                reward_kind=args.reward,
                                with_recorder=False,
                                out_dir=None,
                            )

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
                        out_csv = fold_dir / "test" / "summary_test.csv"
                        df[["r_log_t", "icvar_t", "delta_mdd_t", "reward_t"]].describe().to_csv(out_csv)
                        logger.info(
                            "Online-Eval-Summary geschrieben: %s",
                            out_csv,
                        )

                logger.info("Trial %s erfolgreich abgeschlossen", trial_tag)

            except Exception as e:
                # Trial wurde aufgrund eines Fehlers vorzeitig beendet
                logger.exception(
                    "Trial %s vorzeitig mit Fehler abgebrochen; fahre mit nächstem Trial fort: %s",
                    trial_tag,
                    e,
                )
                continue

        logger.info("Alle Trials abgeschlossen")

    except Exception as e:
        logger.exception("Gesamter Run mit Fehler abgebrochen: %s", e)
        raise
    finally:
        logger.info("Run beendet")


if __name__ == "__main__":
    main()