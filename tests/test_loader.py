import pandas as pd
import pytest
import yaml
from pathlib import Path
from requests import HTTPError
from src.utils.parquet_io import save_parquet, load_parquet

loader = pytest.importorskip("src.data.loader")
load_prices = loader.load_prices

ASSETS_CFG = Path("config/assets_example.yml")
DATA_CFG   = Path("config/data_spec.yml")

@pytest.mark.skipif(not ASSETS_CFG.exists() or not DATA_CFG.exists(), reason="config not found")
def test_smoke():
    cfg = yaml.safe_load(open(ASSETS_CFG))
    spec = yaml.safe_load(open(DATA_CFG))
    df = load_prices(cfg["equities"] + cfg.get("crypto", []),
                     cfg["start"], cfg["end"], spec)
    if df.empty:
        pytest.skip("no data returned")
    assert list(df.index.names) == ["date", "asset"]
    expected_cols = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]
    assert set(df.columns) >= set(expected_cols)
    assert not df.index.duplicated().any()

@pytest.mark.skipif(not DATA_CFG.exists(), reason="data_spec config not found")
def test_invalid_asset(tmp_path):
    with pytest.raises((ValueError, HTTPError)):
        fake_spec = yaml.safe_load(open(DATA_CFG))
        fake_cfg = {"equities": ["FAKE123"], "start": "2020-01-01", "end": "2020-01-10"}
        load_prices(fake_cfg["equities"], fake_cfg["start"], fake_cfg["end"], fake_spec)

@pytest.mark.skipif(not ASSETS_CFG.exists() or not DATA_CFG.exists(), reason="config not found")
def test_persistence(tmp_path):
    cfg = yaml.safe_load(open(ASSETS_CFG))
    spec = yaml.safe_load(open(DATA_CFG))
    df = load_prices(cfg["equities"] + cfg.get("crypto", []),
                     cfg["start"], cfg["end"], spec)
    if df.empty:
        pytest.skip("no data returned")
    file = tmp_path / "tmp.parquet"
    save_parquet(df, file)
    df2 = load_parquet(file)
    assert df.shape == df2.shape
