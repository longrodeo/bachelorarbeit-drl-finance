# Dieses Modul lädt Rohpreisdaten über die Tiingo-API und speichert sie als Parquet.
# Pipeline-Einordnung: Stufe RAW -> unveränderte API-Antworten je Asset persistieren.
# Hauptfunktionen: `_is_crypto` (Tickerklassifikation), `_load_tiingo` (API-Abfrage),
# `download_raw_prices` (Batch-Downloader).
# Eingaben: Asset-Ticker, Zeitfenster (`start`, `end`), optionaler API-Schlüssel.
# Ausgaben: Parquet-Dateien unterhalb des RAW-Verzeichnisses.
# Abhängigkeiten: `pandas` für DataFrames, `requests` für HTTP, interne Pfad-/IO-Helper.
# Edge-Cases: fehlender API-Key, leere oder fehlerhafte Antworten, Netzwerk-Timeouts.
"""
Download von Rohpreisdaten über die Tiingo‑API und Speichern als Parquet.
Stufe: RAW → speichert unveränderte Antworten je Asset.
Unterstützt Aktien und Kryptowährungen; Crypto wird per separatem Endpunkt
abgerufen. Benötigt einen gültigen ``TIINGO_API_KEY``.
Mögliche Fehler: fehlender API‑Key, leere Antworten oder Netzwerkprobleme.
"""

from __future__ import annotations  # ermöglicht Vorwärtsreferenzen in Typannotationen

import os  # Zugriff auf Umgebungsvariablen (API-Key)
from typing import Iterable, Optional, List  # generische Typunterstützung für Sammlungen

import pandas as pd  # Verarbeitung tabellarischer Daten in DataFrames
import requests  # HTTP-Anfragen an Tiingo senden

from src.utils.paths import raw_asset_path, _normalize_asset  # Pfad- und Normalisierungs-Helper
from src.utils.parquet_io import save_parquet  # robustes Parquet-Schreiben

def _is_crypto(asset: str) -> bool:  # prüft auf Krypto-Kürzel anhand USD-Suffix
    """Erkennen, ob ein Ticker eine Krypto‑Notation (z. B. ``BTCUSD``) ist."""
    return asset.upper().endswith("USD")  # Krypto-Paare enden typischerweise auf USD

def _load_tiingo(asset: str, start: str, end: str, token: Optional[str] = None) -> pd.DataFrame:  # API-Aufruf pro Asset
    """Rohdaten direkt von Tiingo laden.

    Parameters
    ----------
    asset : str
        Symbol/Ticker des gewünschten Assets.
    start, end : str
        Zeitfenster im ISO-Format.
    token : str | None
        API-Schlüssel; falls ``None``, wird ``TIINGO_API_KEY`` verwendet.

    Returns
    -------
    pd.DataFrame
        Unveränderte Antwort der Tiingo-API.
    """
    token = token or os.getenv("TIINGO_API_KEY")  # API-Key aus Argument oder Umgebung beziehen
    if not token:  # ohne gültigen Schlüssel lässt sich die API nicht nutzen
        raise RuntimeError("TIINGO_API_KEY is not set.")  # klarer Fehler für fehlende Credentials

    if _is_crypto(asset):  # Branch: Krypto-Ticker benötigen eigenen Endpunkt
        url = "https://api.tiingo.com/tiingo/crypto/prices"  # Basis-URL für Krypto-API
        params = {"tickers": asset.lower(), "startDate": start, "endDate": end, "resampleFreq": "1day", "token": token}  # Query-Parameter zusammenstellen
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()  # GET-Anfrage mit Timeout, Fehler bei HTTP!=200
        payload = r.json()  # Antwort als Python-Struktur dekodieren
        if not payload:  # leere Liste bedeutet keine Daten verfügbar
            raise ValueError(f"No crypto data returned for {asset}.")  # explizite Fehlermeldung
        rows = payload[0].get("priceData", [])  # Preiszeitreihe aus erster Listeneinheit extrahieren
        return pd.DataFrame(rows)  # Umwandlung in DataFrame zur Weiterverarbeitung
    else:  # Branch: klassische Aktien-/ETF-Ticker
        url = f"https://api.tiingo.com/tiingo/daily/{asset}/prices"  # API-Endpunkt je Asset
        params = {"startDate": start, "endDate": end, "resampleFreq": "daily", "token": token}  # Parameter für Tagesdaten
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()  # Abruf End-of-Day Daten, Fehler bei HTTP!=200
        return pd.DataFrame(r.json())  # JSON-Liste direkt in DataFrame konvertieren

def download_raw_prices(assets: Iterable[str], start: str, end: str, token: Optional[str] = None) -> List[str]:  # Batch-Download
    """Mehrere Assets herunterladen und als Parquet speichern.

    Parameters
    ----------
    assets : Iterable[str]
        Liste von Tickern.
    start, end : str
        Zeitfenster der Abfrage.
    token : str | None
        Optionaler API-Key.

    Returns
    -------
    List[str]
        Pfade zu geschriebenen Parquet-Dateien.
    """
    written: List[str] = []  # sammelt Pfade der erfolgreich geschriebenen Dateien
    for asset in assets:  # iteriere über alle angeforderten Ticker
        norm_asset = _normalize_asset(asset)  # Ticker auf API-Konvention normieren
        try:
            df = _load_tiingo(norm_asset, start, end, token=token)  # Einzel-Asset von Tiingo laden
        except Exception as e:  # jegliche Fehler (Netzwerk, API) abfangen
            print(f"[WARN] {asset}: konnte nicht geladen werden ({e}), skip.")  # warnen, aber Pipeline fortsetzen
            continue  # nächstes Asset verarbeiten
        path = raw_asset_path(asset)  # Zielpfad im RAW-Verzeichnis ermitteln
        save_parquet(df, path)  # DataFrame robust als Parquet schreiben
        written.append(str(path))  # Pfad als String in Ergebnisliste aufnehmen
    return written  # Liste aller geschriebenen Dateien zurückgeben
