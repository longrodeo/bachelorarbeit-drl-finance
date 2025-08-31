# scripts/build_state_snapshot.py
from pathlib import Path
import numpy as np
import pandas as pd

from utils.parquet_io import load_parquet, save_parquet
from src.state.states import StateSpec, build_state_for_date

OUT_DIR = Path("data/states_demo")
CLEAN_PATH = Path("data/clean/features_v1.parquet")

def pick_one_date(panel: pd.DataFrame) -> pd.Timestamp:
    dates = panel.index.get_level_values("date").unique().sort_values()
    # Heuristik: nimm einen „reifen“ Tag (nach ~1Y), aber bleib im Bereich
    idx = min(260, max(1, len(dates) - 2))
    return dates[idx]

def to_long_raster(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    """
    X: [C,H,W] = Channels × Features × Assets
    -> long DF: (date, state, channel, feature, asset, value)
    """
    X = state["X"]                  # np.ndarray
    feats = state["features"]       # List[str]
    assets = state["assets"]        # List[str]

    C, H, W = X.shape
    channel_names = ["data"] if C == 1 else ["data", "mask"][:C]

    frames = []
    for ci in range(C):
        df = pd.DataFrame(X[ci], index=feats, columns=assets)
        df = df.stack().reset_index()
        df.columns = ["feature", "asset", "value"]
        df.insert(0, "channel", channel_names[ci] if ci < len(channel_names) else f"ch{ci}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "state", state_name)
    out.insert(0, "date", pd.Timestamp(date))
    return out

def to_scalars_df(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    names = state["g_scalars_names"]
    vals = state["g_scalars"]
    df = pd.DataFrame({"name": names, "value": vals})
    df.insert(0, "state", state_name)
    df.insert(0, "date", pd.Timestamp(date))
    return df

def to_weights_df(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    w = state["g_weights"]
    assets = state["assets"]
    df = pd.DataFrame({"asset": assets, "weight": w})
    df.insert(0, "state", state_name)
    df.insert(0, "date", pd.Timestamp(date))
    return df

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_parquet(CLEAN_PATH)  # MultiIndex (date, asset)
    assert isinstance(panel.index, pd.MultiIndex) and panel.index.nlevels == 2

    # ein Datum wählen
    t = pick_one_date(panel)

    # Assets-Reihenfolge (inkl. CASH)
    assets_order = panel.index.get_level_values("asset").unique().tolist()
    assert "CASH" in assets_order, "CASH muss im CLEAN-Panel enthalten sein."

    # Dummy-Portfolio-Snapshot (nur zum State-Bauen; CASH voll investiert)
    snapshot = {
        "weights": pd.Series(0.0, index=assets_order, dtype=float).set_value("CASH", 1.0)
                  if hasattr(pd.Series, "set_value") else pd.Series(0.0, index=assets_order, dtype=float).rename(None),
        "cash": 1_000_000.0,
        "nav":  1_000_000.0,
        "r_past": 0.0,  # last_portfolio_return
    }
    # Fallback für set_value entfernt: sicherstellen, dass CASH=1.0
    if isinstance(snapshot["weights"], pd.Series):
        snapshot["weights"]["CASH"] = 1.0

    # Spezifikationen
    S0 = StateSpec(
        name="S0",
        per_asset_features=[
            "open","high","low","close",
            "daily_return_log",
            "volatility_becker_parkinson",
            "bid_ask_spread_corwin_schultz",
        ],
        add_mask_channel=True,
    )
    S1 = StateSpec(
        name="S1",
        per_asset_features=[
            # Core
            "open","high","low","close",
            "daily_return_log",
            "volatility_becker_parkinson",
            "bid_ask_spread_corwin_schultz",
            # TA (erweitert)
            "simple_moving_average_20",
            "simple_moving_average_60",
            "exponential_moving_average_12",
            "exponential_moving_average_26",
            "relative_strength_index_14",
            "macd_line_12_26_9",
            "macd_signal_12_26_9",
            "macd_histogram_12_26_9",
            "bollinger_middle_band_20_2.0",
            "bollinger_upper_band_20_2.0",
            "bollinger_lower_band_20_2.0",
            "bollinger_bandwidth_20_2.0",
            "commodity_channel_index_20",
            "average_directional_index_14",
            "positive_directional_index_14",
            "negative_directional_index_14",
        ],
        add_mask_channel=True,
    )

    # States bauen (genau EIN Datum)
    s0 = build_state_for_date(panel, t, S0, assets_order, snapshot, nan_fill_value=0.0)
    s1 = build_state_for_date(panel, t, S1, assets_order, snapshot, nan_fill_value=0.0)

    # Long-Frames erzeugen
    raster_df = pd.concat(
        [to_long_raster(s0, "S0", t), to_long_raster(s1, "S1", t)],
        ignore_index=True,
    )
    scalars_df = pd.concat(
        [to_scalars_df(s0, "S0", t), to_scalars_df(s1, "S1", t)],
        ignore_index=True,
    )
    weights_df = pd.concat(
        [to_weights_df(s0, "S0", t), to_weights_df(s1, "S1", t)],
        ignore_index=True,
    )

    # Speichern (fastparquet)
    save_parquet(raster_df, OUT_DIR / "state_raster_long.parquet")
    save_parquet(scalars_df, OUT_DIR / "state_scalars.parquet")
    save_parquet(weights_df, OUT_DIR / "state_weights.parquet")

    # kleine Konsoleninfo
    print("Datum:", t)
    print("Raster shape S0:", s0["X"].shape, "| S1:", s1["X"].shape)  # [C,H,W]
    print("raster_df rows:", len(raster_df))
    print("scalars_df rows:", len(scalars_df))
    print("weights_df rows:", len(weights_df))
    print("Saved under:", OUT_DIR)

if __name__ == "__main__":
    main()
