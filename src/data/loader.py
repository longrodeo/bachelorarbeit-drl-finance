# src/data/loader.py


from __future__ import annotations
import time
from pathlib import Path
from typing import Iterable
import os, requests
import json, datetime
import numpy as np
import pandas as pd
import yaml

from src.features.core import returns, adv as adv_fn, parkinson_sigma
from src.features.microstructure import (
    cs_beta, cs_gamma, cs_alpha, cs_sigma, cs_spread_from_alpha
)

from src.utils.parquet_io import save_parquet, load_parquet
# Optionaler NYSE-Kalender (sauberer als BusinessDays)
try:
    import exchange_calendars as xcals
    _CAL = xcals.get_calendar("XNYS")
except Exception:
    _CAL = None


# --------------------------- Utilities ---------------------------

def _nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    if _CAL is not None:
        return _CAL.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return pd.bdate_range(start=start, end=end)


def _norm_cols(cols: Iterable) -> list[str]:
    """Spaltennamen in snake_case; bei MultiIndex (tuple) 1. Element nehmen."""
    out = []
    for c in cols:
        if isinstance(c, tuple):
            c = c[0]
        out.append(str(c).lower().replace(" ", "_"))
    return out


# --------------------------- Loader ------------------------------

def _load_tiingo(asset: str, start: str, end: str) -> pd.DataFrame:
    token = os.getenv("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY nicht gesetzt. Bitte als Umgebungsvariable speichern.")

    # --- Asset normalisieren ---
    is_crypto = "-" in asset or asset.lower() in ["btcusd", "ethusd"]
    asset_norm = asset.replace("-", "").lower() if is_crypto else asset

    if is_crypto:
        # ---- Crypto API ----
        url = "https://api.tiingo.com/tiingo/crypto/prices"
        params = {
            "tickers": asset_norm,
            "startDate": start,
            "endDate": end,
            "resampleFreq": "1day",
            "token": token
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data or "priceData" not in data[0]:
            raise ValueError(f"Keine Crypto-Daten von Tiingo für {asset}")
        df = pd.DataFrame(data[0]["priceData"])

    else:
        # ---- Equity/ETF API ----
        url = f"https://api.tiingo.com/tiingo/daily/{asset_norm}/prices"
        params = {
            "token": token,
            "startDate": start,
            "endDate": end,
            "resampleFreq": "daily"
        }
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise ValueError(f"Keine Equity-Daten von Tiingo für {asset}")
        df = pd.DataFrame(data)

    # --- Zeitspalte robust normalisieren ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "time" in df.columns:  # fallback bei Crypto
        df["date"] = pd.to_datetime(df["time"])
    else:
        raise KeyError(f"Keine Zeitspalte in Tiingo-Daten gefunden für {asset}")

    df = df.set_index("date")

    # --- Einheitliche Spalten ---
    mapping = {
        "adjClose": "adj_close",
        "divCash": "dividends",
        "splitFactor": "stock_splits"
    }
    df = df.rename(columns=mapping)

    # Crypto hat kein adj_close → setze = close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # fehlende Spalten ergänzen
    for col in ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]:
        if col not in df.columns:
            df[col] = 0.0

    return df[["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"]]


def load_prices(asset_list: list[str], start: str, end: str, spec: dict) -> pd.DataFrame:

    """
    Lädt OHLCV + Dividends + Stock Splits + Adj Close via yfinance (einzeln je Asset mit Retry).
    Rückgabe: DataFrame mit Index=(date, asset), Spalten=snake_case Felder aus spec['fields'].
    """
    fields = spec["fields"]
    frames: list[pd.DataFrame] = []

    for asset in asset_list:
        # Tiingo als einzige Quelle
        raw = _load_tiingo(asset, start, end)

        if raw.empty:
            print(f"[WARN] Keine Daten für {asset}")
            continue

        keep = [c for c in fields if c in raw.columns]
        df = raw[keep].copy()

        # fehlende Felder (z.B. Dividends/Splits bei Krypto) mit 0 füllen
        for f in fields:
            if f not in df.columns:
                df[f] = 0.0

        df.columns = _norm_cols(df.columns)  # snake_case

        # MultiIndex (date, asset)
        df["asset"] = asset
        df = df.reset_index().set_index(["date", "asset"])
        df.index.set_names(["date", "asset"], inplace=True)
        frames.append(df)

    if not frames:
        raise RuntimeError("Kein Asset erfolgreich geladen (RateLimit oder leere Daten).")

    out = pd.concat(frames).sort_index()
    return out


def align_to_nyse(df: pd.DataFrame, start: str, end: str,
                  crypto_assets: set[str], ffill_crypto: bool = True) -> pd.DataFrame:
    """Align auf NYSE-Handelstage; Krypto optional ffill."""
    sessions = _nyse_sessions(start, end)
    out_frames = []
    for asset, df_a in df.groupby(level="asset"):
        df_a = df_a.droplevel("asset").reindex(sessions)
        if asset in crypto_assets and ffill_crypto:
            df_a = df_a.ffill()
        df_a["asset"] = asset
        df_a = df_a.reset_index().set_index(["index", "asset"])
        df_a.index.set_names(["date", "asset"], inplace=True)
        out_frames.append(df_a)
    out = pd.concat(out_frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]  # Duplikate rausschmeißen
    return out


def add_derived_features(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    win_adv = int(spec["windows"]["adv"])
    win_sigma_hl = int(spec["windows"]["sigma_hl"])
    sl_cs = int(spec["windows"].get("spread_cs", 1))
    rtype = spec.get("return_type", "log")

    out_frames: list[pd.DataFrame] = []
    for asset, x in df.groupby(level="asset"):
        x = x.droplevel("asset").copy()

        # Returns & ADV
        x["return_raw"] = returns(x["close"], kind=rtype)
        x["adv"]        = adv_fn(x["close"], x["volume"], win_adv)

        # Parkinson
        x["sigma_hl"]   = parkinson_sigma(x["high"], x["low"], win_sigma_hl)

        # Corwin–Schultz
        beta  = cs_beta(x["high"], x["low"], sl=sl_cs)
        gamma = cs_gamma(x["high"], x["low"])
        alpha = cs_alpha(beta, gamma)
        x["sigma_cs"]   = cs_sigma(beta, gamma)
        x["spread_cs"]  = cs_spread_from_alpha(alpha)

        # Exec-Referenz (T+1)
        x["exec_ref_tplus1"] = x["open"].shift(-1)

        x["asset"] = asset
        out_frames.append(x.reset_index().set_index(["date", "asset"]))

    df_all = pd.concat(out_frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    return df_all



def _assert_no_dupes(idx: pd.Index) -> None:
    if not idx.is_unique:
        dups = idx[idx.duplicated()].unique()
        raise AssertionError(f"Doppelte Zeitstempel im Index: {len(dups)}")


def _assert_non_negative_prices(df: pd.DataFrame) -> None:
    cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    if (df[cols] < 0).any().any():
        bad = df[(df[cols] < 0).any(axis=1)].index[0]
        raise AssertionError(f"Negative Preise gefunden, erste Zeile: {bad}")


def save_panel(df: pd.DataFrame, cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "data/raw")) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Panel komplett
    save_parquet(df, out_dir / "panel.parquet")

    # Einzel-Assets
    for asset in df.index.get_level_values("asset").unique():
        save_parquet(df.xs(asset, level="asset"), out_dir / f"{asset}.parquet")

    # Manifest schreiben
    manifest = {
        "name": cfg["name"],
        "assets": list(df.index.get_level_values("asset").unique()),
        "shape": df.shape,
        "columns": list(df.columns),
        "created": datetime.datetime.now().isoformat()
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[OK] Panel + Manifest gespeichert unter {out_dir}")
    return out_dir

# --------------------------- CLI -------------------------------

def main(argv=None):
    import argparse

    p = argparse.ArgumentParser("data-loader")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--spec", default=Path("config/data_spec.yml"), type=Path)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))

    start, end = cfg["start"], cfg["end"]
    assets = (cfg.get("equities", []) or []) + (cfg.get("crypto", []) or [])
    crypto_assets = set(cfg.get("crypto", []) or [])

    print("Assets aus der Config:", assets)
    # 1) Download
    df = load_prices(assets, start, end, spec)

    # 2) Align auf NYSE
    df = align_to_nyse(df, start, end, crypto_assets, ffill_crypto=bool(spec["align"]["ffill_crypto"]))

    # 3) Derived Features
    df = add_derived_features(df, spec)

    # 4) Smoke Checks
    _assert_no_dupes(df.index)  # prüft kompletten MultiIndex (date, asset)
    _assert_non_negative_prices(df)

    # 5) Persistenz
    save_panel(df, cfg)

    # Log
    print("Panel geladen:", df.shape)
    print("Spalten:", list(df.columns))
    print("CLI OK — Config:", args.config)
    print("CLI OK — DataSpec:", args.spec)


if __name__ == "__main__":
    main()
