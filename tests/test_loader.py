import os
import pytest
import yaml
from pathlib import Path
from requests import HTTPError

from src.utils.parquet_io import save_parquet, load_parquet
from src.data.load_raw import download_raw_prices
from src.data.build_interim import build_interim_prices

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
def test_smoke_raw_to_interim():
    """RAW → INTERIM läuft durch; INTERIM hat erwarteten Index & Spalten."""
    cfg  = yaml.safe_load(open(ASSETS_CFG, "r", encoding="utf-8"))
    spec = yaml.safe_load(open(DATA_CFG, "r", encoding="utf-8"))

    assets = cfg["equities"] + cfg.get("crypto", [])
    # 1) RAW
    written = download_raw_prices(
        asset_list=assets,
        start=cfg["start"],
        end=cfg["end"],
        out_dir="data/raw",
        provider="tiingo",
    )
    assert len(written) > 0

    # 2) INTERIM
    df = build_interim_prices(
        asset_list=assets,
        start=cfg["start"],
        end=cfg["end"],
        spec=spec,
        raw_dir="data/raw",
        out_path="data/interim/prices.parquet",
        crypto_assets=set(cfg.get("crypto", [])),
    )
    if df.empty:
        pytest.skip("no data returned")

    # MultiIndex + Spalten
    assert list(df.index.names) == ["date", "asset"]
    expected_cols = {
        "open", "high", "low", "close",
        "adj_close", "volume", "dividends", "stock_splits"
    }
    assert expected_cols.issubset(df.columns)
    assert df.index.is_unique


@needs_tiingo
def test_invalid_asset_raises():
    """Ungültiges Asset wirft einen (Provider‑)Fehler."""
    with pytest.raises((ValueError, HTTPError, RuntimeError)):
        download_raw_prices(
            asset_list=["FAKE123"],
            start="2020-01-01",
            end="2020-01-10",
            out_dir="data/raw",
            provider="tiingo",
        )


@needs_cfg
@needs_tiingo
def test_parquet_persistence_roundtrip(tmp_path: Path):
    """Persistenz über parquet_io (fastparquet→pyarrow Fallback)."""
    cfg  = yaml.safe_load(open(ASSETS_CFG, "r", encoding="utf-8"))
    spec = yaml.safe_load(open(DATA_CFG, "r", encoding="utf-8"))
    assets = (cfg["equities"] + cfg.get("crypto", []))[:2]  # klein halten

    # Mini‑Pipeline bis INTERIM
    download_raw_prices(assets, cfg["start"], cfg["end"], out_dir=tmp_path / "raw")
    df = build_interim_prices(
        asset_list=assets,
        start=cfg["start"],
        end=cfg["end"],
        spec=spec,
        raw_dir=tmp_path / "raw",
        out_path=tmp_path / "interim" / "prices.parquet",
        crypto_assets=set(cfg.get("crypto", [])),
    )
    if df.empty:
        pytest.skip("no data returned")

    p = tmp_path / "roundtrip.parquet"
    save_parquet(df, p)
    df2 = load_parquet(p)
    assert df.shape == df2.shape
    assert list(df.columns) == list(df2.columns)
