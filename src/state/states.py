# src/state/builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Literal
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Spec (raster-only, 2-Branch-Design: CNN-Raster + Vektor-Branch)
# ------------------------------------------------------------

@dataclass(frozen=True)
class StateSpec:
    """
    Beschreibt NUR das Raster (per-Asset-Features) für den CNN-Branch.
    Globale Größen (z. B. last_portfolio_return, cash, nav) kommen separat
    als Vektor-Branch zurück.
    """
    name: str
    per_asset_features: List[str]
    add_mask_channel: bool = True          # 2. Kanal mit 1/0 für gültig/NaN
    output_format: Literal["torch_nchw"] = "torch_nchw"
    history: int = 0                       # 0 = Snapshot t (keine Historie)

# optional: YAML laden (falls ihr schon s0/s1.yaml habt)
def load_spec(path: str) -> StateSpec:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML nicht installiert: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return StateSpec(
        name=cfg["name"],
        per_asset_features=list(cfg["per_asset_features"]),
        add_mask_channel=bool(cfg.get("add_mask_channel", True)),
        output_format=cfg.get("output_format", "torch_nchw"),
        history=int(cfg.get("history", 0)),
    )

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def build_state_for_date(
    panel_clean: pd.DataFrame,                 # MultiIndex (date, asset)
    date: pd.Timestamp,
    spec: StateSpec,
    assets_order: List[str],                   # feste Asset-Reihenfolge (inkl. "CASH")
    portfolio_snapshot: Mapping[str, object],  # {"weights": Series|dict, "cash": float, "nav": float, "r_past" or "portfolio_return_prev": float}
    *,
    nan_fill_value: float = 0.0,               # Wert zum Füllen im Datenkanal
) -> Dict[str, object]:
    """
    Baut den State für EIN Datum (2-Branch-Design).

    Rückgabe:
      {
        "X": np.ndarray   # CNN-Raster [C, H, W] = [Channels, Features, Assets]
        "g_scalars": np.ndarray  # Vektor [G] (z. B. last_portfolio_return, cash, nav, ...)
        "g_weights": np.ndarray  # aktuelles Portfolio w_t in assets_order
        "features": List[str],   # Zeilenreihenfolge im Raster (H)
        "assets": List[str],     # Spaltenreihenfolge im Raster (W)
        "date": pd.Timestamp,
        "meta": { ... },
      }
    """
    # --- A) Grundchecks
    assert isinstance(panel_clean.index, pd.MultiIndex) and panel_clean.index.nlevels == 2,\
        "panel_clean muss MultiIndex (date, asset) im Index haben."
    if list(panel_clean.index.names) != ["date", "asset"]:
        # toleranter Slice, aber Hinweis
        pass

    # CASH-Assert (hilfreich, da CASH im Raster erwartet)
    assets_order = list(assets_order)
    assert "CASH" in assets_order, "CASH muss im assets_order enthalten sein."

    # --- B) Slice auf t → DataFrame (index=asset, columns=features)
    try:
        df_t = panel_clean.xs(date, level=0)   # (assets × all_features)
    except KeyError:
        raise KeyError(f"Datum {date} nicht im Panel-Index.")

    # Feature-Auswahl & Ordnung
    missing = [f for f in spec.per_asset_features if f not in df_t.columns]
    if missing:
        raise KeyError(f"Im Panel fehlen Features für Spec {spec.name}: {missing}")

    df_t = df_t.reindex(index=assets_order, columns=spec.per_asset_features)

    # --- C) Raster-Daten [H, W]
    HxW = df_t.T.to_numpy(dtype=float)  # [H, W] = [Features, Assets]

    # Maske 1/0 (gilt für alle Werte)
    mask = (~np.isnan(HxW)).astype(float)

    # Datenkanal NaNs auffüllen
    HxW_filled = HxW.copy()
    HxW_filled[np.isnan(HxW_filled)] = float(nan_fill_value)

    # Channels stapeln
    channels = [HxW_filled[np.newaxis, :, :]]  # [1, H, W]
    if spec.add_mask_channel:
        channels.append(mask[np.newaxis, :, :])  # [1, H, W]
    X = np.vstack(channels)  # [C, H, W]

    if spec.output_format != "torch_nchw":
        raise NotImplementedError(f"output_format {spec.output_format} wird nicht unterstützt.")

    # --- D) Vektor-Branch: globale Größen + Gewichte
    # Gewichtsvektor in assets_order
    g_weights = _vector_per_assets(portfolio_snapshot.get("weights"), assets_order, name="weights")

    # letzte Portfolio-Return-Info (Key tolerant)
    last_ret = portfolio_snapshot.get("portfolio_return_prev", portfolio_snapshot.get("r_past"))
    cash_val = portfolio_snapshot.get("cash")
    nav_val  = portfolio_snapshot.get("nav")

    g_scalars_list: List[float] = []
    g_scalar_names: List[str] = []

    def _push(val, name: str):
        if val is None:
            raise ValueError(f"Portfolio-Snapshot fehlend: {name}")
        g_scalars_list.append(float(val))
        g_scalar_names.append(name)

    _push(last_ret, "last_portfolio_return")
    _push(cash_val, "cash")
    _push(nav_val, "nav")
    g_scalars = np.asarray(g_scalars_list, dtype=float)  # [G]

    # --- E) Zusätzliche Sicherheitsasserts für CASH
    # Falls vorhanden: daily_return_log für CASH sollte nicht NaN sein (Rf-Log)
    if "daily_return_log" in spec.per_asset_features and "CASH" in df_t.index:
        val = df_t.loc["CASH", "daily_return_log"]
        if pd.isna(val):
            raise ValueError("CASH.daily_return_log ist NaN – check CLEAN-Panel/Rf-Ableitung.")

    return {
        "X": X,
        "g_scalars": g_scalars,
        "g_scalars_names": g_scalar_names,
        "g_weights": g_weights,
        "features": list(spec.per_asset_features),
        "assets": assets_order,
        "date": pd.Timestamp(date),
        "meta": {
            "spec": spec.name,
            "num_channels": X.shape[0],
            "num_features": X.shape[1],
            "num_assets": X.shape[2],
        },
    }


def build_states_batch(
    panel_clean: pd.DataFrame,
    dates: List[pd.Timestamp],
    spec: StateSpec,
    assets_order: List[str],
    snapshots: Mapping[pd.Timestamp, Mapping[str, object]],
    *,
    nan_fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Komfort-Funktion für Debug/Pretraining:
    Returns:
      X:          [B, C, H, W]
      g_scalars:  [B, G]
      g_weights:  [B, A]
      batch_dates: List[pd.Timestamp]
    """
    Xs, Gs, Ws = [], [], []
    for t in dates:
        s = build_state_for_date(
            panel_clean=panel_clean, date=t, spec=spec,
            assets_order=assets_order,
            portfolio_snapshot=snapshots[t],
            nan_fill_value=nan_fill_value,
        )
        Xs.append(s["X"])
        Gs.append(s["g_scalars"])
        Ws.append(s["g_weights"])
    X = np.stack(Xs, axis=0)                 # [B, C, H, W]
    g_scalars = np.stack(Gs, axis=0)         # [B, G]
    g_weights = np.stack(Ws, axis=0)         # [B, A]
    return X, g_scalars, g_weights, dates


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _vector_per_assets(weights: object, assets_order: List[str], name: str) -> np.ndarray:
    if weights is None:
        raise ValueError(f"Portfolio-Snapshot enthält keine '{name}'.")
    if isinstance(weights, pd.Series):
        w = weights.reindex(assets_order)
        return w.fillna(0.0).astype(float).to_numpy()
    if isinstance(weights, Mapping):
        return np.array([float(weights.get(a, 0.0)) for a in assets_order], dtype=float)
    raise TypeError(f"'{name}' muss pd.Series oder Mapping sein.")
