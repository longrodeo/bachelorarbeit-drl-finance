import pandas as pd
import numpy as np
from src.features.obs_norm import rolling_zscore

def test_rolling_zscore_basic():
    rng = pd.date_range("2018-01-01", periods=300, freq="B")
    rng_np = np.random.default_rng(0)
    x = pd.Series(rng_np.standard_normal(len(rng)), index=rng, name="feat")
    df = pd.DataFrame({"feat": x})

    z = rolling_zscore(df, window=50)
    # nach Einlaufphase definiert
    assert z.isna().sum().sum() < len(rng)

    # Skalencheck auf dem Tail: ~0-mean, Var grob ~1
    tail = z.iloc[200:]
    m = float(tail.mean().values[0])
    v = float(tail.var().values[0])
    assert abs(m) < 0.1
    assert 0.5 < v < 1.5

def test_rolling_zscore_linear_ramp_level():
    rng = pd.date_range("2018-01-01", periods=300, freq="B")
    x = pd.Series(np.linspace(0, 10, len(rng)), index=rng, name="feat")
    df = pd.DataFrame({"feat": x})
    N = 50
    z = rolling_zscore(df, window=N)

    # Für eine lineare Sequenz ist der Z-Score des letzten Punkts im Fenster
    # nach Einlaufphase (nahezu) konstant:
    # expected = sqrt(3) * (N-1) / sqrt(N^2 - 1)  (bei ddof=0)
    expected = np.sqrt(3) * (N - 1) / np.sqrt(N**2 - 1)
    tail = z.iloc[200:, 0]
    assert tail.std() < 1e-6
    assert np.isclose(tail.iloc[-1], expected, rtol=1e-2, atol=1e-2)
