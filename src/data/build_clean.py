from __future__ import annotations
from typing import Optional
import platform
import numpy as np
import pandas as pd
from pathlib import Path


from src.utils.parquet_io import save_parquet  # euer stabiler IO-Wrapper
from utils.manifest import write_manifest, file_summary, current_commit_short  # :contentReference[oaicite:3]{index=3}

# Feature-Funktionen aus euren Modulen
from src.features.basic_indicator import (
    returns,
    corwin_schultz_beta,
    corwin_schultz_gamma,
    corwin_schultz_alpha,
    becker_parkinson_sigma,     # Volaproxy
    corwin_schultz_spread,      # finaler Spread
)
from src.features.technical_indicators import (
    average_dollar_volume,
    simple_moving_average,
    exponential_moving_average,
    relative_strength_index,
    moving_average_convergence_divergence,
    bollinger,
    commodity_channel_index,
    average_directional_index,
)


def _downcast_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c == "is_cash":
            df[c] = df[c].astype("int8")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int64")
    return df


def _build_cash_asset(
    dates: pd.DatetimeIndex,
    risk_free_annual: pd.Series,
    day_count: int = 360,
    symbol: str = "CASH",
) -> pd.DataFrame:
    """
    Synthetisches, handelbares CASH-Asset:
      open_t  = close_{t-1} (start=1.0)
      close_t = open_t * (1 + r_ann_t * Δ_t/day_count)
      daily_return_log = ln(1 + r_ann_t * Δ_t/day_count)
    """
    rf = risk_free_annual.reindex(dates).ffill()
    date_series = pd.Series(dates, index=dates)
    days_to_next = (date_series.shift(-1) - date_series).dt.days.fillna(1).astype(int)

    factor = 1.0 + rf * (days_to_next / float(day_count))
    factor.iloc[-1] = 1.0  # letzter Tag hat keinen Folgetag

    close = factor.cumprod().astype("float64")
    open_ = close.shift(1).fillna(1.0)
    high = np.maximum(open_.values, close.values)
    low = np.minimum(open_.values, close.values)

    df_cash = pd.DataFrame(
        {
            # Rohschema beibehalten
            "open": open_.values,
            "high": high,
            "low": low,
            "close": close.values,
            "adj_close": close.values,
            "volume": 0.0,
            "dividends": 0.0,
            "stock_splits": 1.0,
            # Core-Features
            "daily_return_log": np.log(factor.values),
            "average_dollar_volume_20": 0.0,
            "volatility_becker_parkinson": 0.0,
            "bid_ask_spread_corwin_schultz": 0.0,
            # TA-Features: nicht sinnvoll für CASH
            "simple_moving_average_20": np.nan,
            "simple_moving_average_60": np.nan,
            "exponential_moving_average_12": np.nan,
            "exponential_moving_average_26": np.nan,
            "relative_strength_index_14": np.nan,
            "macd_line_12_26_9": np.nan,
            "macd_signal_12_26_9": np.nan,
            "macd_histogram_12_26_9": np.nan,
            "bollinger_middle_band_20_2.0": np.nan,
            "bollinger_upper_band_20_2.0": np.nan,
            "bollinger_lower_band_20_2.0": np.nan,
            "bollinger_bandwidth_20_2.0": np.nan,
            "commodity_channel_index_20": np.nan,
            "average_directional_index_14": np.nan,
            "positive_directional_index_14": np.nan,
            "negative_directional_index_14": np.nan,
            # Exec/Flag
            "execution_price_t_plus_1_open": open_.shift(-1).values,
            "is_cash": 1,
        },
        index=dates,
    )
    df_cash["asset"] = symbol
    return df_cash.reset_index().set_index(["date", "asset"])


def build_clean_data(
    prices: pd.DataFrame,
    risk_free_annual: pd.Series,
    out_path: Optional[str] = None,
    cash_symbol: str = "CASH",
    cs_sample_length: int = 1,   # Corwin–Schultz: Spanne (typisch 1–2)
) -> pd.DataFrame:
    """
    Baut ein fixes Feature-Panel inkl. synthetischem CASH-Asset
    und speichert optional via parquet_io.save_parquet(...).

    Erwartet:
      prices: MultiIndex (date, asset), Spalten: open, high, low, close, adj_close, volume, dividends, stock_splits
      risk_free_annual: pd.Series (index=date), annualisierter Tageszins, bereits auf XNYS ff-angepasst
    """
    # Input-Checks
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != ["date", "asset"]:
        raise ValueError("prices muss MultiIndex mit Indexnamen ['date','asset'] besitzen.")
    if cash_symbol in prices.index.get_level_values("asset"):
        raise ValueError(f"Input darf {cash_symbol} noch nicht enthalten.")

    prices = prices.sort_index()
    frames = []

    # --- Nicht-CASH Assets ---
    for asset, df_asset in prices.groupby(level="asset", sort=False):
        px = df_asset.droplevel("asset").sort_index()

        # Core-Features
        daily_ret = returns(px["close"], kind="log")
        adv20 = average_dollar_volume(px["close"], px["volume"], window=20)

        beta = corwin_schultz_beta(px["high"], px["low"], sample_length=cs_sample_length)
        gamma = corwin_schultz_gamma(px["high"], px["low"])
        sigma_bp = becker_parkinson_sigma(beta, gamma)

        alpha = corwin_schultz_alpha(beta, gamma)
        spread_cs = corwin_schultz_spread(alpha)

        # TA-Features
        sma20 = simple_moving_average(px["close"], 20)
        sma60 = simple_moving_average(px["close"], 60)
        ema12 = exponential_moving_average(px["close"], 12)
        ema26 = exponential_moving_average(px["close"], 26)
        rsi14 = relative_strength_index(px["close"], 14)
        macd_line, macd_signal, macd_hist = moving_average_convergence_divergence(px["close"], 12, 26, 9)
        boll_mid, boll_up, boll_lo, boll_bw = bollinger(px["close"], 20, 2.0)
        cci20 = commodity_channel_index(px["high"], px["low"], px["close"], 20)
        adx_df = average_directional_index(px["high"], px["low"], px["close"], 14)

        exec_ref = px["open"].shift(-1)

        features = pd.DataFrame(
            {
                # Rohschema
                "open": px["open"],
                "high": px["high"],
                "low": px["low"],
                "close": px["close"],
                "adj_close": px["adj_close"],
                "volume": px["volume"].astype("float64"),
                "dividends": px["dividends"],
                "stock_splits": px["stock_splits"],

                # Core
                "daily_return_log": daily_ret,
                "average_dollar_volume_20": adv20,
                "volatility_becker_parkinson": sigma_bp,
                "bid_ask_spread_corwin_schultz": spread_cs,

                # Technische Indikatoren
                "simple_moving_average_20": sma20,
                "simple_moving_average_60": sma60,
                "exponential_moving_average_12": ema12,
                "exponential_moving_average_26": ema26,
                "relative_strength_index_14": rsi14,
                "macd_line_12_26_9": macd_line,
                "macd_signal_12_26_9": macd_signal,
                "macd_histogram_12_26_9": macd_hist,
                "bollinger_middle_band_20_2.0": boll_mid,
                "bollinger_upper_band_20_2.0": boll_up,
                "bollinger_lower_band_20_2.0": boll_lo,
                "bollinger_bandwidth_20_2.0": boll_bw,
                "commodity_channel_index_20": cci20,
                "average_directional_index_14": adx_df["adx_14"],
                "positive_directional_index_14": adx_df["plus_di_14"],
                "negative_directional_index_14": adx_df["minus_di_14"],

                # Exec/Flag
                "execution_price_t_plus_1_open": exec_ref,
                "is_cash": 0,
            },
            index=px.index,
        )
        features["asset"] = asset
        frames.append(features.reset_index().set_index(["date", "asset"]))

    # --- CASH Asset ---
    dates = prices.index.get_level_values("date").unique().sort_values()
    cash_df = _build_cash_asset(dates, risk_free_annual, day_count=360, symbol=cash_symbol)
    frames.append(cash_df)

    # --- Zusammenführen, Finalisieren ---
    panel = pd.concat(frames).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    panel = _downcast_feature_dtypes(panel)

    # Optional speichern
    if out_path:
        save_parquet(panel, out_path)

    return panel


def write_clean_manifest(
    spec: dict,
    interim_path: str | Path,
    macro_path: str | Path,
    out_path: str | Path = "data/clean/features_v1.parquet",
    manifest_path: str | Path = "data/clean/_manifest.json",
) -> None:
    """
    Schreibt ein Manifest für die CLEAN/FEATURES-Stufe.
    """
    payload = {
        "stage": "clean",
        "dataset_id": spec.get("feature_version", "v1"),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "git_commit": current_commit_short(),
        "calendar": spec.get("align", {}).get("calendar", "XNYS"),
        "spec": {
            "feature_version": spec.get("feature_version", "v1"),
            "windows": spec.get("windows", {}),
            "cs": spec.get("cs", {}),
            "risk_free": spec.get("risk_free", {}),
            "cash": spec.get("cash", {"symbol": "CASH"}),
        },
        "inputs": [file_summary(str(interim_path)), file_summary(str(macro_path))],
        "outputs": [file_summary(str(out_path))],
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_manifest(payload, str(manifest_path))  # :contentReference[oaicite:4]{index=4}
