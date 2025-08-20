# src/utils/helpers.py
from __future__ import annotations
import logging, os, random
from typing import Optional
import numpy as np

def set_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """
    Setzt Seeds für Python, NumPy und optional PyTorch.
    Macht Hashing reproduzierbar und (optional) Torch deterministisch.
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
    Einheitlicher Logger:
    - StreamHandler (STDOUT) immer.
    - Optional zusätzlicher FileHandler über `to_file`.
    - Mehrfache Handler werden vermieden.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # schon konfiguriert → Level nur aktualisieren
        logger.setLevel(level)
        return logger

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(level)
    logger.addHandler(sh)

    if to_file:
        fh = logging.FileHandler(to_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)

    logger.propagate = False
    logger.setLevel(level)
    return logger
