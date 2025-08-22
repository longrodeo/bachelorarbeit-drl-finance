import os
import pandas as pd
import pytest
import yaml
from pathlib import Path

from src.utils.parquet_io import load_parquet
from src.data.load_raw import download_raw_prices
from src.data.build_interim import build_interim_prices
from src.data.build_clean import build_clean_data

ASSETS_CFG = Path("config/assets_example.yml")
DATA_CFG   = Path("config/data_spec.yml")

needs_cfg = pytest.mark.skipif(
    not ASSETS_CFG.exists() or not DATA_CFG.exists(),
    reason="config not found (config/assets_example.yml, config/data_spec.yml)"
)

needs_tiingo = pytest.mark.skipif(
    not os.getenv("TIINGO_API_KEY"),
    reason="TIINGO_API_KEY not set"
)


@needs_cfg
@needs_tiingo
@pytest.mark.slow
def test_pipeline_raw_interim_clean_runs():
    """End‑to‑End: RAW → INTERIM → CLEAN (mit Features & CASH‑Asset) läuft."""
    cfg  = yaml.safe_load(open(ASSETS_CFG, "r", encoding="utf-8"))
    spec = yaml.safe_load(open(DATA_CFG, "r", encoding="utf-8"))
    assets = cfg["equities"] + cfg.get("crypto", [])

    # RAW
    download_raw_prices(assets, cfg["start"], cfg["end"], out_dir="data/raw")

    # INTERIM
    df_interim = build_interim_prices(
        asset_list=assets,
        start=cfg["start"],
        end=cfg["end"],
        spec=spec,
        raw_dir="data/raw",
        out_path="data/interim/prices.parquet",
        crypto_assets=set(cfg.get("crypto", [])),
    )
    assert not df_interim.empty
    assert df_interim.index.names == ["date", "asset"]

    # Risk‑free Serie (konstant 3% p.a.), auf die INTERIM‑Dates ausgerichtet
    dates = df_interim.index.get_level_values("date").unique().sort_values()
    # tz-naiv für build_clean_data/_build_cash_asset
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    rf = pd.Series(0.03, index=dates)  # 3% annualisiert

    # CLEAN + Features
    out_clean = "data/clean/panel.parquet"
    df_clean = build_clean_data(
        prices=df_interim,
        risk_free_annual=rf,
        out_path=out_clean,
        cash_symbol="CASH",
        cs_sample_length=1,
    )
    assert not df_clean.empty
    assert df_clean.index.names == ["date", "asset"]
    assert Path(out_clean).exists()


@needs_tiingo
@pytest.mark.slow
def test_clean_has_expected_columns():
    """CLEAN enthält Basisfelder + Feature‑Spalten gemäß build_clean_data."""
    p = Path("data/clean/panel.parquet")
    if not p.exists():
        pytest.skip("Bitte zuerst test_pipeline_raw_interim_clean_runs ausführen.")
    df = load_parquet(p)

    base = {
        "open","high","low","close","adj_close","volume","dividends","stock_splits",
        "execution_price_t_plus_1_open","is_cash"
    }
    core = {
        "daily_return_log","average_dollar_volume_20",
        "volatility_becker_parkinson","bid_ask_spread_corwin_schultz"
    }
    ta = {
        "simple_moving_average_20","simple_moving_average_60",
        "exponential_moving_average_12","exponential_moving_average_26",
        "relative_strength_index_14",
        "macd_line_12_26_9","macd_signal_12_26_9","macd_histogram_12_26_9",
        "bollinger_middle_band_20_2.0","bollinger_upper_band_20_2.0","bollinger_lower_band_20_2.0","bollinger_bandwidth_20_2.0",
        "commodity_channel_index_20",
        "average_directional_index_14","positive_directional_index_14","negative_directional_index_14",
    }
    expected = base | core | ta
    missing = expected.difference(df.columns)
    assert not missing, f"Fehlende Spalten: {sorted(missing)}"

    # CASH‑Asset vorhanden
    assert "CASH" in df.index.get_level_values("asset")

    # Index eindeutig
    assert df.index.is_unique

def test_build_clean_fails_if_cash_already_present(tmp_path):
    """Falls im Input schon CASH enthalten ist → ValueError."""
    # Dummy-Daten mit Asset CASH vorbereiten
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    arrays = [dates, ["CASH"] * len(dates)]
    index = pd.MultiIndex.from_arrays(arrays, names=["date", "asset"])
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "adj_close": 1.0,
            "volume": 0.0,
            "dividends": 0.0,
            "stock_splits": 0.0,
        },
        index=index,
    )

    # Risk-free Serie auf denselben Dates
    rf = pd.Series(0.02, index=dates)

    with pytest.raises(ValueError):
        build_clean_data(
            prices=df,
            risk_free_annual=rf,
            out_path=tmp_path / "clean.parquet",
            cash_symbol="CASH",
        )
