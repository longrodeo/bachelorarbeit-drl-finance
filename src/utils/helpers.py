# ---------------------------------------------------------------------------
# Datei: src/utils/helpers.py
# Zweck: Sammlung kleiner Hilfsfunktionen für deterministische Experimente
#   (Seed-Setzung) und einheitliches Logging.
# Hauptfunktionen: ``set_seed`` und ``get_logger``.
# Abhängigkeiten: Standardbibliotheken ``logging``, ``os``, ``random`` sowie
#   ``numpy`` und optional ``torch``.
# Typische Fehler: fehlender PyTorch-Import oder mehrfaches Hinzufügen von
#   Logger-Handlern.
# ---------------------------------------------------------------------------
from __future__ import annotations  # zukünftige Typ-Hints ohne String-Literale
import logging, os, random  # Logging-Framework, Umgebungsvariablen, Zufall
from typing import Optional  # optionaler Parameter-Typ für Pfadangaben
import numpy as np  # numerische Zufallsgeneratoren
from pathlib import Path

def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """
    Setzt Seeds für Python, NumPy und optional PyTorch.
    Hinweis: PYTHONHASHSEED wirkt formal beim Interpreter-Start; das Setzen hier
    hilft v. a. für ggf. gestartete Subprozesse (und schadet nicht).
    
    Parameters
    ----------
    seed : int
        Basiswert für alle Zufallsgeneratoren.
    deterministic_torch : bool, optional
        Erzwingt deterministisches Verhalten bei ``torch`` (CuDNN).
    """
    # Python/OS
    os.environ["PYTHONHASHSEED"] = str(seed)  # Hashseed für Stabilität setzen
    random.seed(seed)  # Python-eigenen PRNG deterministisch machen

    # NumPy
    np.random.seed(seed)  # Numpy-PRNG auf Seed einstellen

    # Torch (optional)
    try:  # Import kann fehlschlagen, wenn Torch nicht installiert ist
        import torch  # schwere Bibliothek für Deep Learning
        torch.manual_seed(seed)  # CPU-Seeds setzen
        torch.cuda.manual_seed_all(seed)  # GPU-Seeds setzen (alle Geräte)
        if deterministic_torch:  # Option für deterministische CuDNN-Läufe
            torch.backends.cudnn.deterministic = True  # deterministische Algorithmen
            torch.backends.cudnn.benchmark = False  # keine autotune-Heuristik
    except ImportError:
        pass  # Torch noch nicht installiert → ignorieren

def get_logger(
    name: str = "BA",
    level: int = logging.INFO,
    *,
    to_file: Optional[str] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Idempotenter Logger:
    - StreamHandler (STDOUT) wird genau einmal sichergestellt.
    - Optional: FileHandler für `to_file` wird genau einmal (pro Pfad) sichergestellt.
    - Keine doppelten Handler; spätere Aufrufe aktualisieren Level/Formatter.
    
    Parameters
    ----------
    name : str
        Logger-Name.
    level : int
        Logging-Level (z. B. ``logging.INFO``).
    to_file : str | None
        Optionaler Pfad für Logdatei.
    fmt : str
        Format-String für Logausgaben.
    datefmt : str
        Datumsformat der Logausgabe.

    Returns
    -------
    logging.Logger
        Konfiguriertes Logger-Objekt ohne doppelte Handler.
    """
    logger = logging.getLogger(name)  # existierenden oder neuen Logger holen
    logger.setLevel(level)  # Mindestlevel setzen
    logger.propagate = False  # keine Weiterleitung an Root-Logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)  # Format definieren

    # 1) StreamHandler sicherstellen (ohne FileHandler—der ist Subklasse von StreamHandler)
    stream_handlers = [  # Filter existierender StreamHandler ohne FileHandler
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    if not stream_handlers:  # falls keiner vorhanden, neu anlegen
        sh = logging.StreamHandler()  # Ausgabe auf STDOUT
        sh.setFormatter(formatter)  # Format zuweisen
        sh.setLevel(level)  # Level setzen
        logger.addHandler(sh)  # Handler anhängen
    else:  # existierende Handler anpassen
        for h in stream_handlers:
            h.setLevel(level)  # Level aktualisieren
            h.setFormatter(formatter)  # Format aktualisieren

    # 2) FileHandler sicherstellen (nur wenn gewünscht und noch nicht vorhanden für genau diesen Pfad)
    if to_file:  # Logging zusätzlich in Datei schreiben
        path = os.path.abspath(to_file)  # absoluter Pfad für Vergleich
        file_handlers = [  # existierende FileHandler mit identischem Pfad suchen
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
        ]
        if not file_handlers:  # wenn noch keiner existiert, neu anlegen
            fh = logging.FileHandler(path, encoding="utf-8")  # Datei-Handler
            fh.setFormatter(formatter)  # Format zuweisen
            fh.setLevel(level)  # Level setzen
            logger.addHandler(fh)  # Handler hinzufügen
        else:  # vorhandene FileHandler aktualisieren
            for h in file_handlers:
                h.setLevel(level)  # Level aktualisieren
                h.setFormatter(formatter)  # Format aktualisieren

    return logger  # fertig konfiguriertes Logger-Objekt zurückgeben

def _jsonable(model, x):
    # primitive
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    # NumPy-Skalare
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
    except Exception:
        pass
    name = getattr(x, "__class__", type(x)).__name__
    # SB3 Schedules (learning_rate etc.)
    if "Schedule" in name or callable(x):
        try:
            return {"type": name, "current": float(model.policy.optimizer.param_groups[0]["lr"])}
        except Exception:
            try:
                return {"type": name, "current": float(x(1.0))}  # initialer LR bei progress=1.0
            except Exception:
                return {"type": name}
    # SB3 TrainFreq
    if name == "TrainFreq":
        n = getattr(x, "n", getattr(x, "frequency", None))
        unit = getattr(x, "unit", None)
        return {"n": int(n) if n is not None else None, "unit": str(unit)}
    # Fallback
    return str(x)

def write_run_manifest(run_dir, algo, model, env, seed, total_timesteps, tensorboard_log=None, deep=False):
    """
    Minimaler, wissenschaftlicher Run-Snapshot als JSON.
    Speichert KEINE großen Artefakte. Optional: Git-Status + Diff.
    """
    import json, sys, platform, subprocess
    import torch, stable_baselines3 as sb3, gymnasium as gym
    from datetime import datetime

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Basis ---
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

    # --- Algo-Hyperparameter (kompakt, robust) ---
    ppo_keys = ["n_steps","batch_size","n_epochs","gamma","gae_lambda","clip_range","ent_coef","vf_coef","max_grad_norm","learning_rate","target_kl"]
    sac_keys = ["batch_size","gamma","tau","train_freq","gradient_steps","learning_starts","buffer_size","ent_coef","learning_rate"]
    keys = ppo_keys if str(algo).lower() == "ppo" else sac_keys
    cfg["algo_kwargs"] = {k: _jsonable(model, getattr(model, k)) for k in keys if hasattr(model, k)}

    # --- Policy/Extractor-Info ---
    try:
        cfg["policy"] = {
            "name": getattr(model, "policy_class", type(getattr(model, "policy", None))).__name__,
            "features_extractor_class": getattr(getattr(model, "policy", None).features_extractor, "__class__", type(None)).__name__,
        }
    except Exception:
        pass

    # --- Spaces & Datenfenster (falls vorhanden) ---
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
    # optional: Reward-/Kosten-Flags, wenn das Env sie anbietet
    for fld in ("reward_kind","fee_bps","vol_targeting"):
        if hasattr(u, fld):
            data_meta[fld] = getattr(u, fld)
    cfg["data"] = data_meta

    # --- (Optional) Git-Infos, rein READ-ONLY ---
    if deep:
        try:
            cfg["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        except Exception:
            cfg["git_commit"] = None
        try:
            cfg["git_dirty"] = bool(subprocess.check_output(["git","status","--porcelain"]).strip())
        except Exception:
            cfg["git_dirty"] = None
        # Diff nur speichern, wenn dirty → verhindert Datenmüll
        if cfg.get("git_dirty"):
            try:
                diff_path = run_dir / f"patch_{cfg.get('git_commit') or 'working'}.diff"
                with open(diff_path, "w", encoding="utf-8") as f:
                    subprocess.run(["git","diff"], stdout=f, check=False)
                cfg["git_diff_file"] = str(diff_path)
            except Exception:
                cfg["git_diff_file"] = None

    # --- Schreiben ---
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
