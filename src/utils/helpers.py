# src/utils/helpers.py
from __future__ import annotations
import logging, os, random
from typing import Optional
import numpy as np

def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """
    Setzt Seeds für Python, NumPy und optional PyTorch.
    Hinweis: PYTHONHASHSEED wirkt formal beim Interpreter-Start; das Setzen hier
    hilft v. a. für ggf. gestartete Subprozesse (und schadet nicht).
    """
    # Python/OS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Torch (optional)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # 1) StreamHandler sicherstellen (ohne FileHandler—der ist Subklasse von StreamHandler)
    stream_handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    if not stream_handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)
    else:
        for h in stream_handlers:
            h.setLevel(level)
            h.setFormatter(formatter)

    # 2) FileHandler sicherstellen (nur wenn gewünscht und noch nicht vorhanden für genau diesen Pfad)
    if to_file:
        path = os.path.abspath(to_file)
        file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path
        ]
        if not file_handlers:
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(level)
            logger.addHandler(fh)
        else:
            for h in file_handlers:
                h.setLevel(level)
                h.setFormatter(formatter)

    return logger
