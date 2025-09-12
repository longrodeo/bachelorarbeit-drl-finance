from pathlib import Path
import pandas as pd

from utils.parquet_io import load_parquet, save_parquet
from state.state_builder import load_spec, build_state_for_date  # YAML→StateSpec + Builder


# Projekt-Root = eine Ebene über "tests"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]   # -> C:\Dev\Bachelorarbeit

# Jetzt alle Pfade relativ zu ROOT
FEATURES_NORM   = ROOT / "data" / "clean" / "features_v1_raw_z.parquet"
ACCOUNT_DIR  = ROOT / "data" / "accounting_demo"
SNAP_PATH    = ACCOUNT_DIR / "portfolio_snapshots.parquet"
REWARD_PATH  = ACCOUNT_DIR / "rewards_log.parquet"
SPEC_S0_YAML = ROOT / "config" / "state_config" / "state0.yml"
SPEC_S1_YAML = ROOT / "config" / "state_config" / "state1.yml"
OUT_DIR      = ROOT / "data" / "states" / "states_demo"
                           # Ausgabe-Ordner
DATE_STR     = "2014-07-14"                                         # gewünschtes Datum (YYYY-MM-DD)

# === Helpers ===================================================================
def to_long_raster(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    """X[C,H,W] → long: (date, state, channel, feature, asset, value)"""
    X = state["X"]                        # np.ndarray [C,H,W]
    feats = state["features"]             # List[str]
    assets = state["assets"]              # List[str]
    C, H, W = X.shape
    ch_names = ["data", "mask"][:C] if C >= 2 else ["data"]

    frames = []
    for ci in range(C):
        df = pd.DataFrame(X[ci], index=feats, columns=assets).stack().reset_index()
        df.columns = ["feature", "asset", "value"]
        df.insert(0, "channel", ch_names[ci] if ci < len(ch_names) else f"ch{ci}")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "state", state_name)
    out.insert(0, "date", pd.Timestamp(date))
    return out

def to_scalars_df(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    names = state["g_scalars_names"]; vals = state["g_scalars"]
    df = pd.DataFrame({"name": names, "value": vals})
    df.insert(0, "state", state_name); df.insert(0, "date", pd.Timestamp(date))
    return df

def to_weights_df(state: dict, state_name: str, date: pd.Timestamp) -> pd.DataFrame:
    w = state["g_weights"]; assets = state["assets"]
    df = pd.DataFrame({"asset": assets, "weight": w})
    df.insert(0, "state", state_name); df.insert(0, "date", pd.Timestamp(date))
    return df



# === Main ======================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Daten laden
    panel = load_parquet(FEATURES_NORM)                       # MultiIndex (date, asset)
    snaps = load_parquet(SNAP_PATH)
    rewards = load_parquet(REWARD_PATH)

    # 1a) Datum in die TZ des Panels bringen
    dates = panel.index.get_level_values("date")
    tz = getattr(dates, "tz", None)

    t_key = pd.Timestamp(DATE_STR)  # "YYYY-MM-DD"
    if tz is not None:
        t_key = (t_key.tz_localize(tz) if t_key.tzinfo is None else t_key.tz_convert(tz)).normalize()
    else:
        t_key = t_key.normalize()

    # 1b) Slice am Datum; wenn der Tag nicht existiert (Feiertag) → nächstgelegene Session
    try:
        df_t = panel.xs(t_key, level="date")  # DataFrame (assets × features)
    except KeyError:
        uniq = dates.unique()
        pos = uniq.get_indexer([t_key], method="nearest")[0]
        t_key = uniq[pos]
        df_t = panel.xs(t_key, level="date")

    # 1c) Risk-free als Skalar aus dem Panel ziehen (Z-Score falls vorhanden)
    rf_col = "risk_free_rate_z" if "risk_free_rate_z" in df_t.columns else "risk_free_rate"
    rf_val = float(df_t[rf_col].iloc[0])

    # 2) Datum wählen
    t = pd.Timestamp(DATE_STR)
    t_date = t.normalize().date()

    # t_plus_1 -> in echtes Datumsfeld überführen (TZ-robust)
    tp1 = pd.to_datetime(snaps["t"], utc=True)  # egal ob ursprünglich naive oder +00:00
    tp1 = tp1.dt.tz_convert(None)  # TZ-Drop
    snaps["_t_plus_1_date"] = tp1.dt.normalize().dt.date

    # 3) Assets-Reihenfolge (inkl. CASH)
    assets_order = panel.index.get_level_values("asset").unique().tolist()


    # 4) Portfolio-Snapshot für genau dieses t:
    #    Wir nehmen die Snapshot-Zeilen, deren t_plus_1 == t (Ende der Runde t-1→t).
    snap_t = snaps.loc[snaps["_t_plus_1_date"] == t_date]
    if snap_t.empty:
        raise ValueError(f"Kein Snapshot gefunden für t_plus_1 == {t.date()} in {SNAP_PATH}")

    # Gewichte (per Asset) aus weight_post_t1, Cash/NAV einmalig (erste Zeile)
    if "asset" not in snap_t.columns or "weight_post_t" not in snap_t.columns:
        raise KeyError("Erwarte Spalten 'asset' und 'weight_post_t1' im Snapshot-Parquet.")
    w_series = (
        snap_t.set_index("asset")["weight_post_t"]
        .reindex(assets_order)
        .fillna(0.0)
        .astype(float)
    )
    cash_val = float(snap_t["cash"].iloc[0]) if "cash" in snap_t.columns else 0.0
    nav_val  = float(snap_t["portfolio_value_t"].iloc[0])

    # letzter Portfolio-Logreturn (r_past) = r_log_t der Runde, deren t_plus_1 == t
    if "r_log_t" in rewards.columns and "t" in rewards.columns:
        # 1) t_plus_1 → naive, normalisierte Timestamps (00:00:00), ohne TZ
        tp1 = pd.to_datetime(rewards["t"], utc=True).dt.tz_convert(None).dt.normalize()
        # 2) t genauso normalisieren (naiv + 00:00:00)
        t_norm = pd.Timestamp(t).tz_localize(None).normalize()
        # 3) Zeile für genau t_plus_1 == t finden (Spalte bleibt unverändert)
        r_past_row = rewards.loc[tp1 == t_norm]
        r_past = float(r_past_row["r_log_t"].iloc[0]) if not r_past_row.empty else 0.0
    else:
        r_past = 0.0  # falls Rewards noch nicht geschrieben sind

    portfolio_snapshot = {
        "weights": w_series,
        "cash": cash_val,
        "nav": nav_val,
        "r_past": r_past,  # alias last_portfolio_return
    }

    t_norm = pd.Timestamp(DATE_STR).tz_localize(None).normalize()


    # 5) Specs aus YAML laden
    S0 = load_spec(str(SPEC_S0_YAML))
    #S1 = load_spec(str(SPEC_S1_YAML))

    # 6) States bauen (genau EIN Datum)
    s0 = build_state_for_date(panel, t, S0, assets_order, portfolio_snapshot, riskfree=rf_val, nan_fill_value=0.0)
    #s1 = build_state_for_date(panel, t, S1, assets_order, portfolio_snapshot, riskfree=rf_val, nan_fill_value=0.0)

    # 7) In Parquets exportieren (long/vektor)
    raster_df0 = to_long_raster(s0, "S0", t)

    scalars_df = to_scalars_df(s0, "S0", t)

    weights_df = to_weights_df(s0, "S0", t)

    #raster_df1= to_long_raster(s1, "S1", t)


    save_parquet(raster_df0, OUT_DIR / "state_raster_long0.parquet")
    save_parquet(scalars_df, OUT_DIR / "state_scalars.parquet")
    save_parquet(weights_df, OUT_DIR / "state_weights.parquet")
    #save_parquet(raster_df1, OUT_DIR / "state_raster_long1.parquet")

    # 8) Konsolen-Check
    print("t =", t)
    #print("S0 X-shape:", s0["X"].shape, "| S1 X-shape:", s1["X"].shape)  # [C,H,W]
    print("raster_df rows:", len(raster_df0))
    print("scalars_df rows:", len(scalars_df))
    print("weights_df rows:", len(weights_df))
   # print("raster_df rows:", len(raster_df1))
    print("saved to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
