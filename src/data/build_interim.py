# Dieses Modul kombiniert RAW-Parquet-Dateien zu einem konsistenten Preis-Panel.
# Pipeline-Einordnung: RAW → INTERIM, dient als sauberer Zwischenschritt für Features.
# Hauptfunktion: `build_interim_prices`; Hilfsfunktionen zur Spalten- und Indexnormierung.
# Eingaben: Asset-Liste, Zeitraum (`start`/`end`), optionale Spezifikation und Krypto-Set.
# Ausgaben: DataFrame mit MultiIndex `(date, asset)`; optional Speicherung auf `INTERIM_PANEL`.
# Abhängigkeiten: `pandas`, Handelskalender/Align-Helper, Parquet-IO.
# Edge-Cases: fehlende Basisspalten, abweichende Spaltennamen, Zeitzonenprobleme, fehlende RAW-Dateien.
"""
Baut aus zuvor geladenen RAW-Dateien ein einheitliches Preis-Panel.
Die Ausgabe besitzt einen MultiIndex ``(date, asset)`` und dient als
Zwischenstufe (INTERIM) für weitere Berechnungen.
Nutzen: Normalisierung verschiedener Datenquellen und striktes Alignen auf
NYSE-Handelstage. Typische Stolperfallen: fehlende Basisspalten oder
Zeitzonenfehler bei den Rohdaten.
"""

from __future__ import annotations  # erlaubt Vorwärtsreferenzen bei Typen
from typing import Optional, Sequence, Set, Dict  # Typinformationen für Argumente

import pandas as pd  # zentrale Datenstruktur (DataFrame)

from src.data.calendar import nyse_trading_days  # Handelskalender für NYSE-Sessions
from src.data.align import align_to_trading_days, resample_crypto_last  # Index-Helfer für Ausrichtung
from src.utils.paths import INTERIM_PANEL  # Zielpfad für das resultierende Panel
from src.utils.parquet_io import load_parquet, save_parquet  # Parquet-Ein-/Ausgabe

# Mapping Provider → kanonisch (nur in INTERIM anwenden)
PROVIDER_TO_CANONICAL = {
    "adjclose": "adj_close",  # Adjusted Close
    "divcash": "dividends",  # Dividendenzahlungen
    "splitfactor": "stock_splits",  # Aktiensplits
}

DEFAULT_SPEC: Dict[str, object] = {
    "fields": ["open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"],  # Standardspalten
    "require_base_fields": True,  # Basisspalten müssen vorhanden sein
    "base_fields": ["open", "high", "low", "close"],  # definierte OHLC-Spalten
    "calendar": "XNYS",  # Standardkalender: NYSE
}

def _sanitize(s: str) -> str:  # Hilfsfunktion zur Normalisierung einzelner Namen
    """Spaltennamen vereinheitlichen (Kleinbuchstaben, Unterstrich)."""
    return str(s).strip().replace(" ", "_").replace("-", "_").lower()  # einfache Normalisierung

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:  # Spaltennamen harmonisieren
    """Spalten auf kanonische Namen mappen."""
    df = df.copy()  # Original unverändert lassen
    # Dict-Comprehension: vorhandene Spalten über PROVIDER_TO_CANONICAL auf Standardnamen abbilden
    df.rename(columns={c: PROVIDER_TO_CANONICAL.get(_sanitize(c), _sanitize(c)) for c in df.columns}, inplace=True)
    return df  # zurückgeben des normierten Frames

def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:  # konvertiert Datumsspalte in Index
    """Datumsspalte in Index umwandeln (tz-naiv)."""
    df = df.copy()  # keine Mutationen am Eingabe-DataFrame
    if "date" in df.columns:  # nur falls Spalte vorhanden
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)  # parse + tz entfernen
        df = df.drop(columns=["date"])  # Originalspalte entfernen
        df.index = dt  # Datumswerte werden Index
        df.index.name = "date"  # Index benennen
    return df  # DataFrame mit Datumsindex zurückgeben

def build_interim_prices(
    assets: Sequence[str],  # Liste der zu verarbeitenden Symbole
    start: str,  # Startdatum des Betrachtungsfensters
    end: str,  # Enddatum des Betrachtungsfensters
    spec: Optional[dict] = None,  # optionale Konfigurationsüberschreibung
    crypto_assets: Optional[Set[str]] = None,  # Menge an Krypto-Tickern
    sessions: Optional[pd.DatetimeIndex] = None,  # vorberechneter Kalender
    save: bool = True,  # Ergebnis schreiben?
) -> pd.DataFrame:  # Rückgabe: Panel-DataFrame
    """RAW-Parquets zu einem kalendarisch ausgerichteten Panel kombinieren.

    Parameters
    ----------
    assets : Sequence[str]
        Liste von zu verarbeitenden Symbolen.
    start, end : str
        Beschränkt das Zeitfenster der Daten.
    spec : dict | None
        Optionale Konfiguration (Spaltenauswahl etc.).
    crypto_assets : Set[str] | None
        Menge an Tickern, die als Krypto behandelt werden.
    sessions : pd.DatetimeIndex | None
        Vorberechneter Kalender; falls ``None`` wird NYSE genutzt.
    save : bool
        Ob das Ergebnis nach ``INTERIM_PANEL`` geschrieben wird.

    Returns
    -------
    pd.DataFrame
        Panel mit Index ``(date, asset)``.
    """
    cfg = dict(DEFAULT_SPEC)  # Start mit Default-Konfiguration
    cal_idx = nyse_trading_days(start, end, tz="UTC")  # Handelskalender erzeugen (UTC, tz-aware)
    if spec:
        cfg.update(spec)  # Nutzerdefinierte Überschreibungen

    fields = list(cfg.get("fields", []))  # gewünschte Spaltenliste
    require_base = bool(cfg.get("require_base_fields", True))  # ob Basisspalten Pflicht sind
    base_fields = set(cfg.get("base_fields", ["open", "high", "low", "close"]))  # Set für schnelle Prüfung
    crypto_assets = {a.upper() for a in (crypto_assets or set())}  # Krypto-Ticker normalisieren

    frames = []  # gesammelt je Asset-DataFrame
    for asset in assets:  # Schleife über alle Assets
        from utils.paths import raw_asset_path  # lazy import, vermeidet Zyklen
        f = raw_asset_path(asset)  # Pfad zur RAW-Datei bestimmen
        if not f.exists():  # Datei muss vorhanden sein
            raise FileNotFoundError(f"RAW file not found: {f}.")  # frühzeitiger Abbruch
        raw = load_parquet(f)  # RAW unverändert (date ist Spalte)

        # Fenster schneiden (tz-naiv)
        if "date" in raw.columns:  # nur falls Datumsspalte existiert
            date = pd.to_datetime(raw["date"], errors="coerce", utc=True).dt.tz_localize(None)  # parse & tz entfernen
            start_date = pd.to_datetime(start); end_date = pd.to_datetime(end)  # Grenzen in datetime umwandeln
            raw = raw.loc[(date >= start_date) & (date <= end_date)]  # Filter auf Zeitfenster

        # Normalisieren
        df = _standardize_columns(raw)  # Spaltennamen vereinheitlichen
        df = _to_datetime_index(df)  # Datumsspalte → Index

        # Feldauswahl
        keep = [c for c in fields if c in df.columns]  # nur benötigte Spalten behalten
        df = df[keep].copy()  # Slice kopieren (Avoid SettingWithCopy)

        # Basisspalten-Policy
        if require_base:  # Validierung aktiv
            missing = [c for c in base_fields if c not in df.columns]  # fehlende OHLC-Felder sammeln
            if missing:
                raise ValueError(f"[{asset}] Missing base fields after mapping: {missing}")  # harte Fehlermeldung

        # sinnvolle Defaults nur falls in fields verlangt
        if "adj_close" in fields and "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]  # Fallback: adj_close = close
        if "dividends" in fields and "dividends" not in df.columns:
            df["dividends"] = 0.0  # fehlende Dividenden auffüllen
        if "stock_splits" in fields and "stock_splits" not in df.columns:
            df["stock_splits"] = 1.0  # neutraler Split-Faktor
        if "volume" in fields and "volume" not in df.columns:
            df["volume"] = 0.0  # fehlendes Volumen = 0

        # Align/Downsample
        is_crypto = (asset.upper() in crypto_assets) or asset.upper().endswith("USD")  # heuristische Krypto-Erkennung
        if is_crypto:
            df = resample_crypto_last(df, cal_idx)  # 7-Tage-Krypto → Handelstage (last)
        else:
            df = align_to_trading_days(df, cal_idx)  # Equities hart auf Handelstage

        df["asset"] = asset  # Asset-Kennung als Spalte
        frames.append(df)  # DataFrame in Liste aufnehmen

    out = pd.concat(frames, axis=0)  # alle Assets untereinander stapeln
    out = out.set_index("asset", append=True).sort_index()  # MultiIndex aufbauen & sortieren
    out.index.set_names(["date", "asset"], inplace=True)  # Indexebenen benennen

    if save:  # optionales Speichern
        save_parquet(out, INTERIM_PANEL)  # Panel persistieren
    return out  # fertiges Panel zurückgeben
