# src/data/bulk_download.py

from __future__ import annotations
import time, math, logging
from pathlib import Path
import yaml
import pandas as pd

from src.data.loader import load_prices, align_to_nyse, add_derived_features, save_panel


def bulk_download(config_path: str | Path, spec_path: str | Path,
                  chunk_size: int = 5, pause_sec: int = 30,
                  log_file: str | None = "bulk_download.log"):
    """
    Lädt alle Assets aus config_path in Blöcken (chunk_size).
    Speichert panel.parquet + Einzelassets.
    """
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8") if log_file else logging.NullHandler()
        ]
    )
    log = logging.getLogger("bulk")

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    spec = yaml.safe_load(open(spec_path, "r", encoding="utf-8"))

    start, end = cfg["start"], cfg["end"]
    assets = (cfg.get("equities", []) or []) + (cfg.get("crypto", []) or [])
    crypto_assets = set(cfg.get("crypto", []) or [])

    n_chunks = math.ceil(len(assets) / chunk_size)
    all_frames = []

    for i in range(n_chunks):
        chunk = assets[i*chunk_size : (i+1)*chunk_size]
        log.info(f"Lade Chunk {i+1}/{n_chunks}: {chunk}")

        try:
            df = load_prices(chunk, start, end, spec)
            df = align_to_nyse(df, start, end, crypto_assets,
                               ffill_crypto=bool(spec["align"]["ffill_crypto"]))
            df = add_derived_features(df, spec)
            all_frames.append(df)
            log.info(f"Chunk {i+1} fertig, Shape={df.shape}")
        except Exception as e:
            log.error(f"Fehler bei Chunk {i+1} ({chunk}): {e}")

        if i < n_chunks-1:
            log.info(f"Pause {pause_sec}s...")
            time.sleep(pause_sec)

    panel = pd.concat(all_frames).sort_index()
    out_dir = save_panel(panel, cfg)
    log.info(f"Bulk-Download abgeschlossen. Panel gespeichert unter {out_dir}")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser("bulk-download")
    p.add_argument("--config", required=True, type=Path, help="Pfad zu assets_*.yml")
    p.add_argument("--spec", default=Path("config/data_spec.yml"), type=Path, help="Pfad zu data_spec.yml")
    p.add_argument("--chunk_size", type=int, default=3, help="Anzahl Assets pro Chunk")
    p.add_argument("--pause_sec", type=int, default=60, help="Pause zwischen Chunks in Sekunden")
    p.add_argument("--log_file", type=Path, default="bulk_download.log", help="Logdatei (default: bulk_download.log)")
    args = p.parse_args(argv)

    bulk_download(args.config, args.spec,
                  chunk_size=args.chunk_size,
                  pause_sec=args.pause_sec,
                  log_file=args.log_file)


if __name__ == "__main__":
    main()
