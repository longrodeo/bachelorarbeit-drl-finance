# tests/test_loader_smoke.py
import pandas as pd
import pytest
from src.data import loader
from src.utils.parquet_io import load_parquet

@pytest.mark.slow
def test_example_loader_runs():
    """Testet ob Example-Config durchläuft und Panel erzeugt wird."""
    import yaml, pathlib

    cfg = yaml.safe_load(open("config/assets_example.yml"))
    spec = yaml.safe_load(open("config/data_spec.yml"))

    df = loader.load_prices(cfg["equities"] + cfg["crypto"],
                            cfg["start"], cfg["end"], spec)
    assert not df.empty
    assert df.index.names == ["date", "asset"]

@pytest.mark.slow
def test_panel_has_expected_columns():
    """Panel enthält alle Rohfelder + Derived Features."""
    df = load_parquet("data/raw/example/panel.parquet")
    expected = {"open","high","low","close","adj_close","volume",
                "dividends","stock_splits",
                "return_raw","adv","sigma_hl","sigma_cs",
                "spread_cs","exec_ref_tplus1"}
    assert expected.issubset(set(df.columns))
    assert df.index.is_unique
