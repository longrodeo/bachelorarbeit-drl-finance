# ---------------------------------------------------------------------------
# Utility helpers for deterministic experiments and consistent logging setup.
# Includes seed initialization across libraries and an idempotent logger factory.
# Also provides utilities to serialize complex objects for manifest snapshots.
# ---------------------------------------------------------------------------
from __future__ import annotations  # allow postponed evaluation of annotations
import logging, os, random  # core modules for logging configuration and entropy
from typing import Optional  # optional type hints for file parameters
import numpy as np  # numerical random number generation utilities
from pathlib import Path

def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Synchronize pseudo random generators for reproducible experiments.

    Args:
        seed: Base value used across Python's random module and NumPy.
        deterministic_torch: When ``True`` the optional torch configuration
            enforces deterministic CuDNN execution, potentially at a speed cost.
    """

    # Align Python's hashing and random module with the provided seed.
    os.environ["PYTHONHASHSEED"] = str(seed)  # ensure deterministic hashing
    random.seed(seed)  # set the global Python PRNG seed

    # Align NumPy's random generator with the same seed for vectorized sampling.
    np.random.seed(seed)

    # Torch is optional; configure it only when available in the environment.
    try:
        import torch  # heavy-weight library for deep learning workloads

        torch.manual_seed(seed)  # set CPU random generator
        torch.cuda.manual_seed_all(seed)  # set GPU generators across devices
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True  # deterministic kernels
            torch.backends.cudnn.benchmark = False  # disable autotune heuristics
    except ImportError:
        pass  # silently skip when torch is not installed

def get_logger(
    name: str = "BA",
    level: int = logging.INFO,
    *,
    to_file: Optional[str] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Provide an idempotent logger with optional file output.

    Args:
        name: Logical logger name used for retrieval and configuration.
        level: Minimum severity accepted by the logger.
        to_file: Optional path to attach a file handler for persistent logs.
        fmt: Formatter string describing the emitted record layout.
        datefmt: Timestamp format applied within the formatter.

    Returns:
        A configured ``logging.Logger`` instance without duplicate handlers.
    """

    logger = logging.getLogger(name)  # retrieve or create the named logger
    logger.setLevel(level)  # enforce the minimum severity threshold
    logger.propagate = False  # avoid double logging through the root logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Ensure exactly one plain StreamHandler that targets stdout.
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    if not stream_handlers:
        sh = logging.StreamHandler()  # primary stream handler for console output
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)
    else:
        for h in stream_handlers:
            h.setLevel(level)
            h.setFormatter(formatter)

    # Attach exactly one file handler per target path whenever requested.
    if to_file:
        path = os.path.abspath(to_file)  # normalize path for comparisons
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
        ]
        if not file_handlers:
            fh = logging.FileHandler(path, encoding="utf-8")  # dedicated file output
            fh.setFormatter(formatter)
            fh.setLevel(level)
            logger.addHandler(fh)
        else:
            for h in file_handlers:
                h.setLevel(level)
                h.setFormatter(formatter)

    return logger

def _jsonable(model, x):
    """Transform configuration values into JSON-serializable representations.

    Args:
        model: Optional model reference used to query dynamic attributes.
        x: Arbitrary object to be serialized when building manifest snapshots.

    Returns:
        A JSON-compatible primitive or structure describing the provided value.
    """

    # Primitives and ``None`` can be returned unchanged.
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x

    # Convert NumPy scalar types to their native Python equivalents.
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass

    name = getattr(x, "__class__", type(x)).__name__  # capture descriptive type name

    # Stable-Baselines3 schedules expose callable learning rate helpers.
    if "Schedule" in name or callable(x):
        try:
            lr_group = model.policy.optimizer.param_groups[0]
            return {"type": name, "current": float(lr_group["lr"])}
        except Exception:
            try:
                return {"type": name, "current": float(x(1.0))}  # evaluate at progress=1.0
            except Exception:
                return {"type": name}

    # Translate SB3 ``TrainFreq`` objects to a concise dictionary.
    if name == "TrainFreq":
        n = getattr(x, "n", getattr(x, "frequency", None))
        unit = getattr(x, "unit", None)
        return {"n": int(n) if n is not None else None, "unit": str(unit)}

    # Fallback to string conversion for unknown objects.
    return str(x)

def write_run_manifest(run_dir, algo, model, env, seed, total_timesteps, tensorboard_log=None, deep=False):
    """Persist a lightweight JSON manifest describing the current training run.

    Args:
        run_dir: Target directory to store the manifest and optional git diff.
        algo: Algorithm identifier or instance name captured for reference.
        model: Trained Stable-Baselines3 model supplying hyperparameters.
        env: Environment instance used during training for metadata extraction.
        seed: Random seed applied to the experiment for reproducibility.
        total_timesteps: Number of timesteps the agent was trained for.
        tensorboard_log: Optional TensorBoard logging directory.
        deep: When ``True`` the manifest includes git metadata and working diff.
    """

    import json, sys, platform, subprocess
    import torch, stable_baselines3 as sb3, gymnasium as gym
    from datetime import datetime

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Core runtime information about environment, device, and versions.
    cfg = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "algo": str(algo),
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "tensorboard_log": tensorboard_log,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "stable_baselines3": sb3.__version__,
            "gymnasium": gym.__version__,
        },
    }

    # Compact hyperparameter snapshot keyed by algorithm-specific attributes.
    ppo_keys = ["n_steps","batch_size","n_epochs","gamma","gae_lambda","clip_range","ent_coef","vf_coef","max_grad_norm","learning_rate","target_kl"]
    sac_keys = ["batch_size","gamma","tau","train_freq","gradient_steps","learning_starts","buffer_size","ent_coef","learning_rate"]
    keys = ppo_keys if str(algo).lower() == "ppo" else sac_keys
    cfg["algo_kwargs"] = {k: _jsonable(model, getattr(model, k)) for k in keys if hasattr(model, k)}

    # Capture policy architecture details when available.
    try:
        cfg["policy"] = {
            "name": getattr(model, "policy_class", type(getattr(model, "policy", None))).__name__,
            "features_extractor_class": getattr(getattr(model, "policy", None).features_extractor, "__class__", type(None)).__name__,
        }
    except Exception:
        pass

    # Collect environment metadata such as spaces and available dataset hints.
    try:
        u = env.unwrapped
    except Exception:
        u = env
    cfg["spaces"] = {
        "observation_space": str(getattr(env, "observation_space", "?")),
        "action_space": str(getattr(env, "action_space", "?")),
    }
    data_meta = {}
    try:
        dates = getattr(u, "dates", None)
        if dates is not None and len(dates) > 0:
            data_meta["dates_start"] = str(dates[0])
            data_meta["dates_end"]   = str(dates[-1])
    except Exception:
        pass
    try:
        port = getattr(u, "portfolio", None)
        if port is not None:
            data_meta["n_assets"] = int(getattr(port, "n_assets", 0))
    except Exception:
        pass
    for fld in ("reward_kind","fee_bps","vol_targeting"):
        if hasattr(u, fld):
            data_meta[fld] = getattr(u, fld)
    cfg["data"] = data_meta

    # Optionally append git state for reproducibility and audit purposes.
    if deep:
        try:
            cfg["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        except Exception:
            cfg["git_commit"] = None
        try:
            cfg["git_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"]).strip())
        except Exception:
            cfg["git_dirty"] = None
        if cfg.get("git_dirty"):
            try:
                diff_path = run_dir / f"patch_{cfg.get('git_commit') or 'working'}.diff"
                with open(diff_path, "w", encoding="utf-8") as f:
                    subprocess.run(["git","diff"], stdout=f, check=False)
                cfg["git_diff_file"] = str(diff_path)
            except Exception:
                cfg["git_diff_file"] = None

    # Persist the manifest configuration to disk in a stable JSON format.
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
