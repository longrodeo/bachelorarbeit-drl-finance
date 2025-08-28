"""
Integrationstests für Loader und Parquet-Persistenz.
Prüfen Datenpipeline von RAW bis INTERIM sowie Fehlerpfade und
Roundtrip-Speicherung.
"""

# Testmodul für die Daten-Download- und Build-Pipeline inklusive
# Parquet-Speicherungen.
# Nutzt TIINGO-API (via API-Key) sowie YAML-Konfigurationsdateien.
# Validiert, dass Daten von RAW über INTERIM ohne Fehler fließen.
# Persistenz wird über einen Roundtrip-Test abgesichert.
# Edge-Cases: fehlende Konfigurationen, ungültige Symbole, Engine-Fallbacks.

import os  # Zugriff auf Umgebungsvariable TIINGO_API_KEY
import pytest  # PyTest-Framework
import yaml  # YAML-Konfigurationsdateien laden
from pathlib import Path  # Pfadobjekte für Konfig-Dateien
from requests import HTTPError  # Exception-Typ des HTTP-Clients

# Parquet-Lese/Schreib-Helfer
from src.utils.parquet_io import save_parquet, load_parquet
# Downloader für Rohdaten
from src.data.load_raw import download_raw_prices
# Builder für Zwischenpanel
from src.data.build_interim import build_interim_prices

# Pfade zu Assets- und Dataspec-Konfiguration
ASSETS_CFG = Path("config/assets_regions.yml")
DATA_CFG   = Path("config/data_spec.yml")

needs_cfg = pytest.mark.skipif(
    not ASSETS_CFG.exists() or not DATA_CFG.exists(),
    reason="config not found (config/assets_example.yml, config/data_spec.yml)",
)  # ohne Konfigurationen → Tests überspringen

needs_tiingo = pytest.mark.skipif(
    not os.getenv("TIINGO_API_KEY"),
    reason="TIINGO_API_KEY not set",
)  # API-Key erforderlich

@needs_cfg
@needs_tiingo
# Happy-path: komplette Pipeline RAW→INTERIM.
def test_smoke_raw_to_interim():
    """RAW → INTERIM läuft durch; INTERIM hat erwarteten Index & Spalten."""
    cfg  = yaml.safe_load(open(ASSETS_CFG, "r", encoding="utf-8"))  # Assets laden
    spec = yaml.safe_load(open(DATA_CFG, "r", encoding="utf-8"))  # Spezifikation laden
    assets = cfg["equities"] + cfg.get("crypto", [])  # kombinierte Asset-Liste
    start = spec["window"]["start"]  # Startdatum aus Spec
    end = spec["window"]["end"]  # Enddatum aus Spec

    # 1) RAW
    written = download_raw_prices(assets, start, end)  # Daten herunterladen
    assert len(written) > 0  # mindestens eine Datei geschrieben

    # 2) INTERIM
    df = build_interim_prices(
    assets, start, end,
    spec = spec,
    crypto_assets = set(cfg.get("crypto", [])),
    save = False,
    )  # DataFrame für INTERIM erzeugen

    if df.empty:
        pytest.skip("no data returned")  # Keine Daten: Test überspringen

    # MultiIndex + Spalten prüfen
    assert list(df.index.names) == ["date", "asset"]  # Indexstruktur
    expected_cols = {
        "open", "high", "low", "close",
        "adj_close", "volume", "dividends", "stock_splits"
    }
    assert expected_cols.issubset(df.columns)  # erforderliche Spalten vorhanden
    assert df.index.is_unique  # keine doppelten (date, asset)-Paare


@needs_tiingo
# Fehlerhafte Assets sollen stillschweigend übersprungen werden.
def test_invalid_asset_is_skipped():
    """Ungültiges Asset wird vom Loader übersprungen (kein Raise, leere Rückgabe)."""
    written = download_raw_prices(["FAKE123"], "2020-01-01", "2020-01-10")  # bewusst falsches Symbol
    assert written == []  # Downloader liefert leere Liste statt Exception


@needs_cfg
@needs_tiingo
# Persistenz der Daten über Parquet-IO sicherstellen.
def test_parquet_persistence_roundtrip(tmp_path: Path):
    """Persistenz über parquet_io (fastparquet→pyarrow Fallback)."""
    cfg  = yaml.safe_load(open(ASSETS_CFG, "r", encoding="utf-8"))  # Konfiguration laden
    spec = yaml.safe_load(open(DATA_CFG, "r", encoding="utf-8"))  # Spezifikation laden
    assets = (cfg["equities"] + cfg.get("crypto", []))[:2]  # kleines Asset-Subset
    start = spec["window"]["start"]
    end = spec["window"]["end"]


    # Mini‑Pipeline bis INTERIM
    download_raw_prices(assets, start, end)  # RAW herunterladen
    df = build_interim_prices(
    assets, start, end,
    spec = spec,
    crypto_assets = set(cfg.get("crypto", [])),
    save = False,
    )  # INTERIM erzeugen

    if df.empty:
        pytest.skip("no data returned")  # Skip bei fehlenden Daten

    p = tmp_path / "roundtrip.parquet"  # temporären Parquet-Pfad setzen
    save_parquet(df, p)  # DataFrame speichern
    df2 = load_parquet(p)  # wieder einlesen
    assert df.shape == df2.shape  # Shape unverändert
    assert list(df.columns) == list(df2.columns)  # Spalten identisch

