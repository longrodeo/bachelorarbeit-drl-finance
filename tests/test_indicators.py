# tests/test_indicators_reference.py
import numpy as np
import pandas as pd
import pytest

# Referenz-Lib (skip, wenn nicht installiert)
pta = pytest.importorskip("pandas_ta")      # -> import pandas_ta as pta


# Dein Modul
import src.features.technical_indicators as ti


# ---------- Fixtures & Helpers ----------

@pytest.fixture(scope="module")
def ohlcv():
    """
    Erzeugt deterministische OHLCV-Zeitreihen (Business Days) als Grundlage
    für alle Tests. So vermeiden wir Abhängigkeiten von externen Dateien.
    """
    rng = pd.date_range("2014-06-02", "2015-02-01", freq="B", tz="UTC")
    rs = np.random.RandomState(42)

    price = 100 + rs.normal(0, 1, len(rng)).cumsum()
    high  = pd.Series(price + np.abs(rs.normal(0.5, 0.2, len(rng))), index=rng, name="high")
    low   = pd.Series(price - np.abs(rs.normal(0.5, 0.2, len(rng))), index=rng, name="low")
    close = pd.Series(price + rs.normal(0, 0.1, len(rng)), index=rng, name="close")
    volume= pd.Series(np.exp(rs.normal(12, 0.5, len(rng))), index=rng, name="volume")  # lognormal ~ "realistisch"

    # Open = lagged close (nur als plausible Größe)
    open_ = close.shift(1).fillna(close).rename("open")

    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def assert_series_close(a, b, tol=1e-8, min_overlap=0.9, label="", burn_in=0):
    a = pd.Series(a, dtype=float)
    b = pd.Series(b, dtype=float)
    if burn_in:
        a = a.iloc[burn_in:]
        b = b.iloc[burn_in:]
    na, nb = a.dropna(), b.dropna()
    idx = na.index.intersection(nb.index)
    if len(idx) == 0:
        raise AssertionError(f"{label} keine Überlappung.")
    overlap = len(idx) / max(1, min(len(na), len(nb)))
    if overlap < min_overlap:
        raise AssertionError(f"{label} Overlap {overlap:.1%} < {min_overlap:.1%}.")
    diff = (a.loc[idx]-b.loc[idx]).abs()
    i, md = diff.idxmax(), float(diff.max())
    msg = (f"{label} max|Δ|={md:.3e} @ {i} (tol={tol:.2e}); "
           f"median|Δ|={diff.median():.3e}, mean|Δ|={diff.mean():.3e}, n={len(idx)}.\n"
           f"Top5:\n{diff.nlargest(5).to_string()}")
    assert md <= tol, msg




# ---------- Tests: einfache Indikatoren ----------

def test_sma_matches_pandasta(ohlcv):
    close = ohlcv["close"]
    ours = ti.simple_moving_average(close, 20)
    ref  = pta.sma(close, length=20)  # pandas-ta accessor
    assert_series_close(ours, ref, tol=1e-10)


def test_ema_matches_pandasta(ohlcv):
    close = ohlcv["close"]
    ours = ti.exponential_moving_average(close, span=20)
    ref  = pta.ema(close, length=20)
    assert_series_close(ours, ref, tol=2e-2, burn_in=60, label="ema20")


def test_adv_matches_definition(ohlcv):
    close, volume = ohlcv["close"], ohlcv["volume"]
    ours = ti.average_dollar_volume(close, volume, window=20)
    ref  = (close * volume).rolling(20).mean()
    assert_series_close(ours, ref, tol=1e-12)


# ---------- Tests: RSI / MACD / Bollinger ----------

def test_rsi_wilder_matches_pandasta(ohlcv):
    close = ohlcv["close"]
    ours = ti.relative_strength_index(close, period=14)     # Wilder via ewm(alpha=1/period)
    ref  = pta.rsi(close, length=14)                          # pandas_ta default: Wilder
    # Hinweis: In der Warm-up-Phase liefern beide NaN → Vergleich ignoriert die NaNs.
    assert_series_close(ours, ref, tol=6e-2, burn_in=70, label="rsi14")


def test_macd_matches_pandasta(ohlcv):
    close = ohlcv["close"]
    macd, signal, hist = ti.moving_average_convergence_divergence(close, 12, 26, 9)
    ref = pta.macd(close, fast=12, slow=26, signal=9)
    # pandas_ta-Konventionen
    burn = 5 * max(12, 26, 9)
    assert_series_close(macd, ref["MACD_12_26_9"], tol=5e-4, burn_in=burn, label="macd")
    assert_series_close(signal, ref["MACDs_12_26_9"], tol=6e-4, burn_in=burn, label="macd_signal")
    assert_series_close(hist, ref["MACDh_12_26_9"], tol=5e-4, burn_in=burn, label="macd_hist")


def test_bollinger_matches_pandasta(ohlcv):
    close = ohlcv["close"]
    mid, upper, lower, width = ti.bollinger(close, window=20, n_std=2.0)
    ref = pta.bbands(close, length=20, std=2.0)
    # pandas_ta-Spalten: BBL_20_2.0 (unteres), BBM_20_2.0 (mittleres), BBU_20_2.0 (oberes)
    burn = 3 * 20
    assert_series_close(mid, ref["BBM_20_2.0"], tol=1e-6, burn_in=burn, label="bb_mid")
    assert_series_close(upper, ref["BBU_20_2.0"], tol=1e-6, burn_in=burn, label="bb_up")
    assert_series_close(lower, ref["BBL_20_2.0"], tol=1e-6, burn_in=burn, label="bb_lo")
    # width ist abgeleitet; optional Plausibilitätscheck
    assert (width.dropna() >= 0).all()


# ---------- Tests: CCI / ADX ----------

def test_cci_matches_pandasta(ohlcv):
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    ours = ti.commodity_channel_index(h, l, c, period=20)
    ref  = pta.cci(h, l, c, length=20)  # nutzt 0.015-Skalierung
    assert_series_close(ours, ref, tol=1e-6, min_overlap=0.8)


def test_adx_matches_pandasta(ohlcv):
    h, l, c = ohlcv["high"], ohlcv["low"], ohlcv["close"]
    ref = pta.adx(h, l, c, length=14)
    ours = ti.average_directional_index(h, l, c, period=14)
    # pandas_ta: ADX_14, DMP_14 (= +DI), DMN_14 (= -DI)
    burn = 5 * 14
    assert_series_close(ours["adx_14"], ref["ADX_14"], tol=2e-1, burn_in=burn, label="adx")
    assert_series_close(ours["plus_di_14"], ref["DMP_14"], tol=2e-1, burn_in=burn, label="+di")
    assert_series_close(ours["minus_di_14"], ref["DMN_14"], tol=2e-1, burn_in=burn, label="-di")


# ---------- Sanity: NaN-Warmup-Längen ----------

def test_nan_warmup_lengths_are_reasonable(ohlcv):
    close = ohlcv["close"]; h=ohlcv["high"]; l=ohlcv["low"]
    # SMA 20: erste 19 NaN
    sma20 = ti.simple_moving_average(close, 20)
    assert sma20.isna().sum() >= 19

    # RSI 14: Anfang NaN, danach begrenzt (hier keine exakte Zahl geprüft, nur "Warm-up existiert")
    rsi14 = ti.relative_strength_index(close, 14)
    # 1) Wertebereich muss passen
    assert rsi14.dropna().between(0, 100).all()
    # 2) Warm-up: entweder echte NaNs (min_periods=period) ODER sofortige Werte (seeded)
    na = rsi14.isna().sum()
    assert (na >= 14) or (na == 0)

    # MACD: Warm-up existiert
    macd, signal, hist = ti.moving_average_convergence_divergence(close, 12, 26, 9)
    na_macd = macd.isna().sum()
    na_signal = signal.isna().sum()
    assert ((na_macd > 0 and na_signal > 0) or (na_macd == 0 and na_signal == 0))

