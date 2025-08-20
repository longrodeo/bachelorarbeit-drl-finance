# tests/test_helpers.py
import os
import numpy as np

def test_set_seed_numpy_repro():
    from src.utils.helpers import set_seed
    set_seed(123)
    a = np.random.RandomState(0).rand(3)  # unabhängig: nur check, dass keine Exceptions oben
    # reproduzierbare globale RNG:
    set_seed(123)
    x1 = np.random.rand(5)
    set_seed(123)
    x2 = np.random.rand(5)
    assert np.allclose(x1, x2)

def test_logger_basic(tmp_path):
    from src.utils.helpers import get_logger
    log_file = tmp_path / "run.log"
    logger = get_logger("TEST", to_file=str(log_file))
    logger.info("hello")
    # Handler nicht doppelt:
    logger2 = get_logger("TEST")
    assert logger is logger2
    # File wurde beschrieben:
    s = log_file.read_text(encoding="utf-8")
    assert "hello" in s
