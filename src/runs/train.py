# src/runs/train.py
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import shutil, json

import pandas as pd

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.utils.helpers import set_seed, get_logger
import src.state.state_builder as sb
from src.env.data_builder import load_data_for_windows, build_env_segment
from src.data.parquet_io import load_parquet
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
    env = build_env_segment(
        panel, seg, state_spec=spec, reward_kind=reward, with_recorder=False, out_dir=None
    )
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
        fns = [
            (lambda s=seed_base + i: _make_seeded_env(panel, base_seg, spec, reward, s))
            for i in range(n_envs)
        ]
    else:  # "segments": Segmente ggf. zyklisch auffüllen bis n_envs erreicht
        if n_envs <= len(train_segs):
            use_segs = train_segs[:n_envs]
        else:
            reps = (n_envs + len(train_segs) - 1) // len(train_segs)
            use_segs = (train_segs * reps)[:n_envs]
        fns = [
            (lambda seg=use_segs[i], s=seed_base + i: _make_seeded_env(panel, seg, spec, reward, s))
            for i in range(n_envs)
        ]
    Vec = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return Vec(fns)


def _load_trials(path):
    import yaml, json

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def _read_synth_splits(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenarios = []
    if isinstance(cfg, dict) and "scenarios" in cfg:
        scenarios = cfg["scenarios"] or []
    elif isinstance(cfg, list):
        scenarios = cfg
    else:
        scenarios = []

    out = []
    for sc in scenarios:
        if not isinstance(sc, dict) or "file" not in sc:
            continue
        name = sc.get("name") or Path(sc["file"]).stem
        out.append({"name": name, "file": sc["file"]})
    return out


def _segment_for_panel(panel: pd.DataFrame) -> tuple[str, str]:
    dates = panel.index.get_level_values(0).unique()
    dates = pd.to_datetime(dates)

    # tz-aware -> tz-naive
    try:
        if getattr(dates, "tz", None) is not None:
            dates = dates.tz_convert(None)
    except Exception:
        # some indexes behave differently; best effort
        pass

    start = pd.Timestamp(dates.min()).date().isoformat()
    end = pd.Timestamp(dates.max()).date().isoformat()
    return start, end

def _validate_synth_panel(panel: pd.DataFrame, spec) -> None:
    # 1) Index-Check
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Synth-Panel muss einen MultiIndex (date, asset) haben.")
    if list(panel.index.names) != ["date", "asset"]:
        raise ValueError(
            f"Synth-Panel index.names muss ['date','asset'] sein, ist aber {panel.index.names}. "
            "Fix: index umbenennen und als MultiIndex speichern."
        )

    # 2) Pflichtspalten für Env (rf_* wird in build_env_segment() immer gezogen)
    required_cols = ["rf_daily_factor_raw", "risk_free_rate_raw"]
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            "Synth-Panel fehlt Pflichtspalten für Risk-Free: "
            f"{missing}. Du musst die Pfade erst durch run_data_pipeline.py --build_synth schicken."
        )

    # 3) Asset-Universum muss enthalten sein (sonst scheitert TradingEnv beim Slicing)
    from src.utils.paths import get_assets_flat, get_asset_groups

    required_assets = set(get_assets_flat(get_asset_groups()))
    have_assets = set(panel.index.get_level_values("asset").unique())
    missing_assets = sorted(required_assets - have_assets)
    if missing_assets:
        raise ValueError(
            "Synth-Panel enthält nicht alle benötigten Assets aus assets_regions. "
            f"Fehlend: {missing_assets}"
        )

    # 4) Feature-Spalten aus state_spec müssen vorhanden sein
    per_asset = list(getattr(spec, "per_asset_features", []) or [])
    global_f = list(getattr(spec, "global_features", []) or [])
    mask_f = getattr(spec, "mask_feature", None)

    needed = per_asset + global_f + ([mask_f] if mask_f else [])
    missing_feat = [f for f in needed if f and f not in panel.columns]
    if missing_feat:
        raise ValueError(
            "Synth-Panel fehlt Feature-Spalten aus state_spec: "
            f"{missing_feat}. Lösung: build_clean.py/run_data_pipeline.py auf Synth-Pfade anwenden."
        )


def _rel_to_data_dir(p: Path) -> Path:
    # compute_rewards_from_snapshots expects accounting_dir relative to project/data
    try:
        return p.relative_to(paths.DATA_DIR)
    except Exception:
        parts = p.parts
        if parts and parts[0].lower() == "data":
            return Path(*parts[1:])
        try:
            return p.relative_to(Path.cwd() / "data")
        except Exception as e:
            raise RuntimeError(f"Kann {p} nicht relativ zu project/data umwandeln") from e


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["standard", "train_and_test_synth"], default="standard")

    # Standard-Modus (CPCV/WF) benötigt splits; für synth-mode nicht
    p.add_argument("--strategy", choices=["cpcv", "walkforward"], default="cpcv")
    p.add_argument("--wf_mode", choices=["refit", "warm"], default="refit")  # nur für walkforward
    p.add_argument("--splits", default=None)

    # Synth-Mode Inputs (Training-Zeitraum + Szenario-Liste)
    p.add_argument("--train_start", default="2015-01-01")
    p.add_argument("--train_end", default="2024-12-31")
    p.add_argument("--synth_splits", default="config/splits/splits_synth.yaml")
    p.add_argument("--synth_subset", default=None,
                   help="Komma-separierte Liste von Szenario-Namen aus synth_splits.yaml (z.B. 'bear_1y,side_lowvol_1y'). Default=None -> alle.",
    )

    p.add_argument("--state_spec", required=True)  # z.B. config/state_config/state0.yml
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

    if args.mode == "standard":
        if not args.splits:
            p.error("--splits ist erforderlich im mode=standard")
    set_seed(args.seed)

    # Run-Ordner und Logger vorbereiten
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy_tag = args.strategy if args.mode == "standard" else "synth"
    run_dir = (
        Path(args.run_root)
        / f"{args.algo}_{args.reward}_{Path(args.state_spec).stem}_{strategy_tag}_{ts}"
    )
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
            "mode": args.mode,
            "strategy": args.strategy,
            "wf_mode": args.wf_mode,
            "splits": args.splits,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "synth_splits": args.synth_splits,
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
            "splits_file": str(args.splits) if args.splits else None,
            "state_spec": str(args.state_spec),
            "algo": args.algo,
            "reward": args.reward,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # copy config files for audit
        if args.hpo_mode == "grid" and args.hpo_param_file:
            try:
                shutil.copyfile(args.hpo_param_file, run_dir / Path(args.hpo_param_file).name)
            except Exception as e:
                logger.warning("Konnte hpo_param_file nicht kopieren: %s", e)

        if args.mode == "standard" and args.splits:
            try:
                shutil.copyfile(args.splits, run_dir / Path(args.splits).name)
            except Exception as e:
                logger.warning("Konnte splits-Datei nicht kopieren: %s", e)

        if args.mode == "train_and_test_synth" and args.synth_splits:
            try:
                shutil.copyfile(args.synth_splits, run_dir / Path(args.synth_splits).name)
            except Exception as e:
                logger.warning("Konnte synth_splits-Datei nicht kopieren: %s", e)

        # ---- Splits / Windows bauen ----
        if args.mode == "standard":
            windows = list(_read_windows(args.strategy, args.splits))
            logger.debug("[_read_windows] returned type: %s", type(windows))
            logger.info("[RUN] %d splits loaded from %s", len(windows), args.splits)

            # Beispiel: nur diese Folds in Stage A verwenden
            if args.hpo_splits:
                keep = [int(x) for x in args.hpo_splits.split(",")]
                windows = [windows[i] for i in keep]
                logger.info("HPO_splits aktiv: %s -> %d verwendete Folds", args.hpo_splits, len(windows))
        else:
            # 1 Fold: Train auf kompletter Range, keine "historischen" Testsegmente (Tests passieren auf Synth)
            windows = [
                {
                    "train": [(args.train_start, args.train_end)],
                    "test": [],
                }
            ]
            logger.info(
                "[SYNTH] 1 Fold erzeugt: Train=(%s..%s), Tests=3 Szenarien aus %s",
                args.train_start,
                args.train_end,
                args.synth_splits,
            )

        # Panel laden
        load_strategy = args.strategy if args.mode == "standard" else "cpcv"
        panel = load_data_for_windows(
            windows,
            strategy=load_strategy,
            features_source=args.features_source,
        )
        spec = sb.load_spec(str(args.state_spec))
        logger.info("Panel und state_spec geladen")

        if args.mode == "standard":
            if args.strategy.lower() == "walkforward":
                logger.info("[DATA] walk-forward -> using features_source=%s", args.features_source)
            else:
                years = sorted(
                    {int(s[0][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])}
                    | {int(s[1][:4]) for w in windows for part in ("train", "test") for s in w.get(part, [])}
                )
                logger.info("[DATA] cpcv -> loading per-year parquet for years=%s", years)
        else:
            logger.info("[DATA] synth-mode -> loaded historical years %s..%s", args.train_start[:4], args.train_end[:4])

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

                    # Synth-mode: immer "refit" Semantik (neu bauen/set_env egal, es gibt eh nur 1 fold)
                    do_refit = (args.mode == "train_and_test_synth") or (
                        args.strategy == "cpcv"
                        or (args.strategy == "walkforward" and args.wf_mode == "refit")
                    )

                    if do_refit:
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

                    reset_steps = not (args.strategy == "walkforward" and args.wf_mode == "warm")
                    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, reset_num_timesteps=reset_steps)
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
                        logger.debug("rollout_size/batch_size-Check übersprungen (fehlende Attribute)")

                    train_vec.close()

                    # ============================================================
                    # TEST: Standard (CPCV/WF) oder Synth (3 Szenarien)
                    # ============================================================

                    if args.mode == "standard":
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

                                obs = _unpack_reset(env)
                                done = False
                                while not done:
                                    action, _ = model.predict(obs, deterministic=True)
                                    obs, reward, done, info = _unpack_step(env.step(action))
                                env.close()

                                acc_dir_rel = _rel_to_data_dir(seg_dir)

                                compute_rewards_from_snapshots(
                                    accounting_dir=acc_dir_rel,
                                    spec=RewardSpec(kind=args.reward),
                                    out_name="rewards.parquet",
                                )

                        # ===== TEST: online =====
                        else:
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
                            logger.info("Online-Eval-Summary geschrieben: %s", out_csv)

                    else:
                        # ============ SYNTH TEST (3 Szenarien) ============
                        scenarios = _read_synth_splits(args.synth_splits)
                        if not scenarios:
                            raise RuntimeError(f"Keine Szenarien in synth_splits gefunden: {args.synth_splits}")

                        if args.synth_subset:
                            want = [x.strip() for x in args.synth_subset.split(",") if x.strip()]
                            scenarios = [sc for sc in scenarios if sc["name"] in set(want)]
                            if not scenarios:
                                raise RuntimeError(f"--synth_subset matcht kein Szenario. Gewünscht={want}")

                        for sc in scenarios:
                            sc_name = sc["name"]
                            sc_file = sc["file"]

                            logger.info(
                                "Synth-Test: Trial %s, Fold %02d, Szenario=%s, File=%s",
                                trial_tag,
                                k,
                                sc_name,
                                sc_file,
                            )

                            synth_panel = load_parquet(sc_file)
                            _validate_synth_panel(synth_panel, spec)
                            seg = _segment_for_panel(synth_panel)

                            sc_dir = fold_dir / "test_synth" / sc_name
                            sc_dir.mkdir(parents=True, exist_ok=True)

                            (sc_dir / "scenario_meta.json").write_text(
                                json.dumps({"name": sc_name, "file": sc_file, "segment": seg}, indent=2),
                                encoding="utf-8",
                            )

                            if args.eval_mode == "from_snapshots":
                                env = build_env_segment(
                                    synth_panel,
                                    seg,
                                    state_spec=spec,
                                    reward_kind=args.reward,
                                    with_recorder=True,
                                    out_dir=sc_dir,
                                )

                                obs = _unpack_reset(env)
                                done = False
                                while not done:
                                    action, _ = model.predict(obs, deterministic=True)
                                    obs, reward, done, info = _unpack_step(env.step(action))
                                env.close()

                                acc_dir_rel = _rel_to_data_dir(sc_dir)

                                compute_rewards_from_snapshots(
                                    accounting_dir=acc_dir_rel,
                                    spec=RewardSpec(kind=args.reward),
                                    out_name="rewards.parquet",
                                )
                            else:
                                online = OnlineEvaluator(kind=args.reward)
                                env = build_env_segment(
                                    synth_panel,
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

                                out_csv = sc_dir / "summary_test_synth.csv"
                                df[["r_log_t", "icvar_t", "delta_mdd_t", "reward_t"]].describe().to_csv(out_csv)
                                logger.info("Synth online summary geschrieben: %s", out_csv)

                logger.info("Trial %s erfolgreich abgeschlossen", trial_tag)

            except Exception as e:
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
