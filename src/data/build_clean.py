# -----------------------------------------------------------------------------
# Constructs the CLEAN feature panel by enriching INTERIM prices with technical
# indicators, synthesising a cash asset, and persisting data alongside a
# metadata manifest for downstream consumption.
# -----------------------------------------------------------------------------

"""Build the CLEAN stage feature panel and accompanying metadata artifacts."""

from __future__ import annotations  # allow forward references in type hints
from typing import Optional, Hashable
import platform  # provide runtime metadata for the manifest
import pandas as pd
from pathlib import Path

from src.utils.parquet_io import save_parquet  # resilient Parquet writer with engine fallbacks
from src.utils.manifest import write_manifest, file_summary, current_commit_short  # manifest utilities

from src.features.basic_indicator import (
    returns,
    corwin_schultz_beta,
    corwin_schultz_gamma,
    corwin_schultz_alpha,
    becker_parkinson_sigma,
    corwin_schultz_spread_sanitized,
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
    """Downcast numeric feature columns to memory-friendly dtypes."""
    for c in df.columns:  # inspect every column individually
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")  # compress floating point features to 32 bit
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int64")  # ensure integer-like columns use int64 for stability
    return df  # return DataFrame with adjusted dtypes


def _fmt_label(k: Hashable) -> str:
    """Normalise an asset label into a string representation."""
    return "/".join(map(str, k)) if isinstance(k, tuple) else str(k)


def build_clean_data(
    prices: pd.DataFrame,
    out_path: Optional[str] = None,
    cs_sample_length: int = 2,  # Corwin–Schultz estimation window (typically 1–2)
) -> pd.DataFrame:
    """Create the CLEAN feature panel with technical indicators and cash asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Input panel indexed by (date, asset) containing price information.
    out_path : str | None
        Optional path to persist the resulting feature panel.
    cs_sample_length : int
        Window length for the Corwin–Schultz spread estimation.

    Returns
    -------
    pd.DataFrame
        Fully populated feature panel including the synthetic cash asset.
    """
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != ["date", "asset"]:
        raise ValueError("prices must be a MultiIndex with index names ['date','asset'].")

    prices = prices.sort_index()  # ensure chronological ordering across all assets
    frames = []  # collects per-asset DataFrames prior to concatenation

    # --- Non-cash assets ---
    for asset, df_asset in prices.groupby(level="asset", sort=False):  # iterate by asset symbol
        px = df_asset.droplevel("asset").sort_index()  # obtain the single-asset time series

        # Core features
        daily_ret = returns(px["adj_close"], kind="log")  # logarithmic returns per session
        adv20 = average_dollar_volume(px["close"], px["volume"], window=20)  # liquidity proxy

        beta = corwin_schultz_beta(px["high"], px["low"], sample_length=cs_sample_length)  # spread proxy component
        gamma = corwin_schultz_gamma(px["high"], px["low"])
        sigma_bp = becker_parkinson_sigma(beta, gamma)  # volatility estimate from high/low range

        alpha = corwin_schultz_alpha(beta, gamma)

        CRYPTO_BASES = {"BTC-USD", "ETH-USD"}

        def _is_crypto_label(asset) -> bool:
            """Detect whether the asset label corresponds to a crypto pair."""
            s = "/".join(map(str, asset)) if isinstance(asset, tuple) else str(asset)
            base = s.split("-")[0].upper()
            return base in CRYPTO_BASES or s.endswith("-USD")

        if _is_crypto_label(asset):
            # Conservative static crypto trading cost assumption (30 bps)
            spread_cs = pd.Series(0.0030, index=px.index, dtype=float)
        else:
            # Sanitised Corwin–Schultz spread with rolling smoothing and floor
            spread_cs = corwin_schultz_spread_sanitized(alpha, roll=5, floor=1e-4)

        # Debug helper retained for reference on negative alpha ratios
        """neg = int((alpha <= 0).sum())
        tot = int(alpha.shape[0])
        print(f"[CS] {asset}: alpha<=0 = {neg}/{tot} ({neg / tot:.1%})")"""

        # Technical indicator features
        sma20 = simple_moving_average(px["close"], 20)  # short-term trend
        sma60 = simple_moving_average(px["close"], 60)  # medium-term trend
        ema12 = exponential_moving_average(px["close"], 12)  # fast exponential average
        ema26 = exponential_moving_average(px["close"], 26)  # slow exponential average
        rsi14 = relative_strength_index(px["close"], 14)  # momentum oscillator
        macd_line, macd_signal, macd_hist = moving_average_convergence_divergence(px["close"], 12, 26, 9)
        boll_mid, boll_up, boll_lo, boll_bw = bollinger(px["close"], 20, 2.0)  # Bollinger bands and bandwidth
        cci20 = commodity_channel_index(px["high"], px["low"], px["close"], 20)
        adx_df = average_directional_index(px["high"], px["low"], px["close"], 14)

        features = pd.DataFrame(
            {
                # Raw schema
                "open": px["open"],  # session open price
                "high": px["high"],  # session high price
                "low": px["low"],    # session low price
                "close": px["close"],  # session close price
                "adj_open": px["adj_open"],  # adjusted open price
                "adj_close": px["adj_close"],  # adjusted close price
                "volume": px["volume"].astype("float64"),  # trading volume
                "dividends": px["dividends"],  # cash dividends paid
                "stock_splits": px["stock_splits"],  # split factor

                # Core metrics
                "daily_return_log": daily_ret,  # log return per session
                "average_dollar_volume_20": adv20,  # 20-day average dollar volume
                "volatility_becker_parkinson": sigma_bp,  # volatility proxy
                "bid_ask_spread_corwin_schultz": spread_cs,  # spread estimate

                # Technical indicators
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
            },
            index=px.index,
        )
        features.index.name = "date"
        features = features.assign(asset=asset)  # add asset identifier column
        features = features.set_index("asset", append=True)
        frames.append(features)  # accumulate MultiIndex (date, asset) frames

    normed = []
    for f in frames:
        if isinstance(f, pd.Series):
            normed.append(f.to_frame(f.name or "value"))
        else:
            normed.append(f)

    # --- Combine and finalise ---
    panel = pd.concat(normed).sort_index()  # concatenate all assets together
    panel = panel[~panel.index.duplicated(keep="last")]  # drop duplicate rows conservatively
    panel = _downcast_feature_dtypes(panel)  # optimise dtype footprint

    # Optional persistence of the feature panel
    if out_path:
        save_parquet(panel, out_path)  # persist to disk using robust writer

    return panel  # return the assembled feature panel


def write_clean_manifest(
    spec: dict,
    interim_path: str | Path,
    macro_path: str | Path,
    out_path: str | Path = "data/clean/features_v1.parquet",
    manifest_path: str | Path = "data/clean/_manifest.json",
) -> None:
    """Persist metadata manifest describing the CLEAN stage outputs.

    Parameters
    ----------
    spec : dict
        Configuration used during feature generation.
    interim_path, macro_path : Path | str
        Source datasets that fed into CLEAN creation.
    out_path : Path | str
        Location of the generated feature panel.
    manifest_path : Path | str
        Destination path for the manifest JSON file.
    """
    payload = {
        "stage": "clean",  # pipeline stage identifier
        "dataset_id": spec.get("feature_version", "v1"),  # feature version label
        "created_at": pd.Timestamp.utcnow().isoformat(),  # creation timestamp
        "git_commit": current_commit_short(),  # repository commit reference
        "calendar": spec.get("align", {}).get("calendar", "XNYS"),  # trading calendar used
        "spec": {
            "feature_version": spec.get("feature_version", "v1"),
            "windows": spec.get("windows", {}),
            "cs": spec.get("cs", {}),
            "risk_free": spec.get("risk_free", {}),
        },
        "inputs": [file_summary(str(interim_path)), file_summary(str(macro_path))],  # upstream sources
        "outputs": [file_summary(str(out_path))],  # produced artifacts
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_manifest(payload, str(manifest_path))  # write manifest JSON to disk
