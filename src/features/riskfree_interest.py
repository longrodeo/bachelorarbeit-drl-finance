# ---------------------------------------------------------------------------
# Datei: src/features/riskfree_interest.py
# Zweck: Abruf und Aufbereitung risikofreier Tageszinsen über FRED.
# Hauptfunktionen: ``fetch_fred_nyse_daily`` sowie Umrechnungen in Tagesraten.
# Ein-/Ausgabe: FRED-Serien-ID → ``pd.Series`` ausgerichtet auf NYSE-Kalender.
# Abhängigkeiten: ``requests``, ``pandas``, interne Kalender/Align-Helfer.
# Edge Cases: fehlender API-Key, leere Antwort, Zeitzonenabgleich.
# ---------------------------------------------------------------------------

"""Download und Aufbereitung risikofreier Zinsen (FRED-API).
Lieferant: Federal Reserve Bank (z. B. Treasury-Renditen). Die Daten werden auf
NYSE-Handelstage ausgerichtet und optional aufgefüllt."""

import os  # Zugriff auf Umgebungsvariablen für API-Key
from typing import Optional  # optionale Parameterannotationen
import requests  # HTTP-Anfragen an FRED-Server
import pandas as pd  # Datenhaltung und Transformation

from src.data.calendar import nyse_trading_days  # Handelskalender mit Feiertagen
from src.data.align import align_to_trading_days  # Reindex-Helfer für Series

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"  # Basis-Endpoint der API

def _resolve_fred_api_key(passed: Optional[str] = None) -> str:
    """FRED API-Key ermitteln.

    Reihenfolge der Suche:
    1. explizit übergebenes ``passed``
    2. Umgebungsvariablen ``FRED_API_KEY``, ``FRED_API_TOKEN`` oder ``FRED_KEY``

    Parameters
    ----------
    passed : str, optional
        Bereits bekannter API-Key.

    Returns
    -------
    str
        Gefundener API-Key.
    """
    # key = "Platzhalter"  # exemplarischer Platzhalter (nicht genutzt)
    key = passed or os.environ.get("FRED_API_KEY") or os.environ.get("FRED_API_TOKEN") or os.environ.get("FRED_KEY")  # schrittweise Auflösung
    if not key:  # falls kein Key ermittelt wurde
        raise ValueError("FRED API Key fehlt. Setze FRED_API_KEY (oder übergib api_key=...).")  # eindeutige Fehlermeldung
    return key  # zurückgeben des Strings


def fetch_fred_nyse_daily(
    series_id: str = "DGS3MO",
    start: str = "1990-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    fill: str = "ffill",   # "ffill", "bfill" oder None
    tz: str = "UTC",
) -> pd.Series:
    """Zeitreihe von FRED holen und auf NYSE-Sessions reindizieren.

    Parameters
    ----------
    series_id : str, optional
        FRED-Serienkennzeichen, Standard ``DGS3MO`` (3-Monats-Treasury).
    start, end : str, optional
        ISO-Datum der Abfragegrenzen.
    api_key : str, optional
        API-Key, wird sonst via ``_resolve_fred_api_key`` gesucht.
    fill : str, optional
        Auffüllmethode (``ffill``/``bfill``/``None``).
    tz : str, optional
        Zeitzone des Zielindex (Standard UTC).

    Returns
    -------
    pd.Series
        Serie mit Tageszinsen ausgerichtet auf NYSE-Handelstage.
    """
    api_key = _resolve_fred_api_key(api_key)  # API-Key beschaffen

    if end is None:  # falls kein Enddatum angegeben wurde
        end = pd.Timestamp.today(tz="UTC").date().isoformat()  # heutiges Datum in ISO-Form

    # --- FRED abrufen ---
    params = {  # Parameter für HTTP-GET
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "file_type": "json",
        "api_key": api_key,
    }
    resp = requests.get(FRED_URL, params=params, timeout=30)  # GET-Request mit Timeout
    resp.raise_for_status()  # Fehler werfen, falls Status != 200
    data = resp.json()  # JSON-Antwort parsen
    obs = pd.DataFrame(data.get("observations", []))  # Observations in DataFrame
    if obs.empty:  # keine Daten zurückbekommen
        return pd.Series(name=series_id, dtype="float64")  # leere Serie

    # --- in Series (Prozent p.a.) ---
    obs["value"] = pd.to_numeric(obs["value"].replace(".", pd.NA), errors="coerce")  # Prozentwerte in float umwandeln
    obs["date"]  = pd.to_datetime(obs["date"], utc=True).dt.tz_localize(None)  # Datum ohne Zeitzone
    s = pd.Series(obs["value"].values, index=obs["date"].values, name=series_id).sort_index()  # sortierte Series

    # --- NYSE-Handelstage  + Reindex  ---
    cal_idx = nyse_trading_days(start=s.index.min().date().isoformat(), end=end, tz=tz)  # Handelskalender erzeugen
    df = align_to_trading_days(s.to_frame(), cal_idx)  # Reindex auf Kalender

    if fill == "ffill":  # vorne auffüllen
        df[series_id] = df[series_id].ffill()
    elif fill == "bfill":  # hinten auffüllen
        df[series_id] = df[series_id].bfill()

    return df[series_id]  # Series mit Tageszinsen

def annual_pct_to_daily_rate(y_annual_pct: pd.Series, basis: int = 360) -> pd.Series:
    """Prozentangaben p.a. in tägliche einfache Rate (dezimal) umwandeln.

    Parameters
    ----------
    y_annual_pct : pd.Series
        Jahresrenditen in Prozent.
    basis : int, optional
        Zinstagebasis (360 oder 365).

    Returns
    -------
    pd.Series
        Tageszinssätze in Dezimalform.
    """
    return (y_annual_pct.astype(float) / 100.0) / float(basis)  # Prozent → Dezimal / Basis

def daily_factor(y_annual_pct: pd.Series, basis: int = 360) -> pd.Series:
    """Multiplikativen Faktor ``1 + r_d`` aus Jahresprozenten ableiten.

    Parameters
    ----------
    y_annual_pct : pd.Series
        Jahresrendite in Prozent.
    basis : int, optional
        Zinstagebasis.

    Returns
    -------
    pd.Series
        ``1 + r_d`` als Faktor zur Multiplikation.
    """
    return 1.0 + annual_pct_to_daily_rate(y_annual_pct, basis=basis)  # Aufschlag auf 1
