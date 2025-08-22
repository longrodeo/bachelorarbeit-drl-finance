from __future__ import annotations
from pathlib import Path
import platform
import pandas as pd

from src.data.calendar import nyse_trading_days
from src.data.align import align_to_trading_days, resample_crypto_last
from src.utils.parquet_io import load_parquet, save_parquet
from utils.manifest import write_manifest, file_summary, current_commit_short  # :contentReference[oaicite:1]{index=1}

__all__ = ["build_interim_prices", "write_interim_manifest"]


def build_interim_prices(
    asset_list: list[str],
    start: str,
    end: str,
    spec: dict,
    raw_dir: str | Path = "data/raw",
    out_path: str | Path = "data/interim/panel.parquet",
    crypto_assets: set[str] | None = None,
) -> pd.DataFrame:
    """
    Nimmt RAW-Dateien, wählt Felder (spec['fields']), normalisiert Spaltennamen (snake_case),
    richtet auf NYSE aus und schreibt EIN kombiniertes Parquet (MultiIndex: date, asset) nach data/interim/.
    - Fehlende optionale Felder werden sinnvoll ergänzt (dividends=0.0, stock_splits=1.0, adj_close=close, volume=0.0).
    - Keine Fills in INTERIM (Lücken bleiben sichtbar). Krypto-Downsampling via resample_crypto_last.
    """
    crypto_assets = crypto_assets or set()
    fields: list[str] = list(spec.get("fields", []))
    if not fields:
        raise ValueError("spec['fields'] muss eine Liste gültiger Feldnamen enthalten.")

    cal_idx = nyse_trading_days(start, end)  # Master-Kalender
    frames: list[pd.DataFrame] = []
    rawp = Path(raw_dir)

    for asset in asset_list:
        f = rawp / f"{asset}.parquet"
        if not f.exists():
            print(f"[WARN] RAW fehlt: {f}")
            continue

        raw = load_parquet(f)

        # Sicherstellen: DatetimeIndex, Name 'date'
        if not isinstance(raw.index, pd.DatetimeIndex):
            if "date" in raw.columns:
                raw = raw.set_index("date")
        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "date"

        # Feldauswahl (nur vorhandene)
        keep = [c for c in fields if c in raw.columns]
        df = raw[keep].copy()

        # --- Harte Prüfung: unerlässliche Preisspalten müssen vorhanden sein
        required_price = {"open", "high", "low", "close"}
        missing_req = [c for c in required_price if c not in df.columns]
        if missing_req:
            raise ValueError(f"[{asset}] Fehlende Preisspalten: {missing_req} in RAW {f}")

        # --- Fehlende optionale Felder sinnvoll ergänzen
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]
        if "dividends" not in df.columns:
            df["dividends"] = 0.0
        if "stock_splits" not in df.columns:
            df["stock_splits"] = 1.0
        if "volume" not in df.columns:
            df["volume"] = 0.0

        # snake_case
        df.columns = [c.replace(" ", "_").replace("-", "_").lower() for c in df.columns]

        # Alignment
        if asset in crypto_assets:
            df_aligned = resample_crypto_last(df, cal_idx)
        else:
            df_aligned = align_to_trading_days(df, cal_idx)

        # MultiIndex aufbauen
        df_aligned["asset"] = asset
        df_aligned = df_aligned.reset_index().set_index(["date", "asset"])
        frames.append(df_aligned)

    if not frames:
        raise RuntimeError("Keine INTERIM-Frames erzeugt (RAW leer/fehlend?).")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(out, outp)
    print(f"[OK] INTERIM gespeichert: {outp} (rows={len(out)})")
    return out


def write_interim_manifest(
    spec: dict,
    raw_files: list[str] | list[Path],
    out_path: str | Path = "data/interim/panel.parquet",
    manifest_path: str | Path = "data/interim/_manifest.json",
) -> None:
    """
    Schreibt ein Manifest für die INTERIM-Stufe.
    """
    payload = {
        "stage": "interim",
        "dataset_id": spec.get("interim_dataset_id", "panel_v1"),
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "git_commit": current_commit_short(),
        "calendar": spec.get("align", {}).get("calendar", "XNYS"),
        "spec": {
            "fields": spec.get("fields", []),
            "align": {"ffill_crypto": bool(spec.get("align", {}).get("ffill_crypto", False))}
        },
        "inputs": [file_summary(str(p)) for p in raw_files],
        "outputs": [file_summary(str(out_path))],
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_manifest(payload, str(manifest_path))  # :contentReference[oaicite:2]{index=2}
