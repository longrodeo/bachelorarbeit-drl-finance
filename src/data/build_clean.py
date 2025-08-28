# ---------------------------------------------------------------------------
# Datei: src/data/build_clean.py
# Zweck: Kombiniert das INTERIM-Panel mit technischen Indikatoren und einem
#   synthetischen risikofreien Asset ("CASH"). Ergebnis ist die CLEAN-Stufe.
# Hauptfunktionen: ``_build_cash_asset`` erzeugt das Kunst-Asset, ``build_clean_data``
#   berechnet Features und vereinigt alles, ``write_clean_manifest`` schreibt
#   Metadaten.
# Ein-/Ausgabe: MultiIndex-Panel ``(date, asset)`` → erweitertes Feature-Panel
#   sowie optionale Parquet/Manifest-Dateien.
# Abhängigkeiten: ``pandas``, ``numpy`` sowie eigene Feature-Module; Stolpersteine
#   sind falsche Datentypen oder bereits vorhandenes CASH-Asset.
# ---------------------------------------------------------------------------
"""
Erzeugt das finale Feature-Panel (CLEAN) inklusive synthetischem CASH-Asset.
Verknüpft Preisdaten mit technischen Indikatoren und speichert optional
Parquet-Dateien sowie ein Manifest. Abhängigkeiten reichen von NumPy/Pandas bis
zu eigenen Feature-Modulen. Typische Fehler: falsche Datentypen oder bereits
vorhandenes CASH-Asset im Input.
"""

# ``from __future__`` erlaubt spätere Typreferenzen ohne String-Literale
from __future__ import annotations
# ``Optional``-Alias für optionale Parameter bei Manifest/Output-Pfaden
from typing import Optional
# Metainformationen über Python-Version usw. für Manifest
import platform  # Versionsinfo fürs Manifest
import numpy as np  # numerische Operationen
import pandas as pd  # Datenverarbeitung
from pathlib import Path  # Pfad-Manipulation


# Stabiler Parquet-Schreiber mit Engine-Fallbacks
from src.utils.parquet_io import save_parquet  # stabiler IO-Wrapper
# Manifest-Helfer für Prüfsummen und Commit-Referenzen
from utils.manifest import write_manifest, file_summary, current_commit_short  # Manifest-Helfer

# Feature-Funktionen aus euren Modulen
from src.features.basic_indicator import (
    returns,
    corwin_schultz_beta,
    corwin_schultz_gamma,
    corwin_schultz_alpha,
    becker_parkinson_sigma,     # Volaproxy
    corwin_schultz_spread,      # finaler Spread
)
# Technische Indikatoren zur Trend-/Volatilitätsanalyse
from src.features.technical_indicators import (
    average_dollar_volume,            # Umsatzbasierte Liquidität
    simple_moving_average,            # Gleichgewichteter gleitender Mittelwert
    exponential_moving_average,       # EMA mit stärkerem Gewicht auf jüngste Werte
    relative_strength_index,          # Momentum-Oszillator (0-100)
    moving_average_convergence_divergence,  # Trendfolger mit zwei EMAs
    bollinger,                        # Bänder auf Basis SMA und StdAbw
    commodity_channel_index,          # Abweichung vom gleitenden Mittel
    average_directional_index,        # Trendstärke via +DI/-DI
)


def _downcast_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Speicherfreundliche Datentypen für Feature-Spalten setzen."""
    for c in df.columns:  # jede Spalte einzeln prüfen
        if c == "is_cash":
            df[c] = df[c].astype("int8")  # Flag benötigt nur wenige Bits
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")  # Float‑Features auf 32 Bit
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int64")  # int64 für Mengen/Volumen
    return df  # DataFrame mit optimierten Datentypen zurückgeben


def _build_cash_asset(
    dates: pd.DatetimeIndex,
    risk_free_annual: pd.Series,
    day_count: int = 360,
    symbol: str = "CASH",
 ) -> pd.DataFrame:

    """Synthetisches CASH-Asset auf Basis risikofreier Tageszinsen.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Handelstage, auf denen das Asset notiert.
    risk_free_annual : pd.Series
        Jahreszins (dezimal) pro Handelstag.
    day_count : int
        Zins-Basis, meist 360.
    symbol : str
        Name des künstlichen Assets.

    Returns
    -------
    pd.DataFrame
        Preis- und Feature-Schema für das CASH-Asset.
    """
    rf = risk_free_annual.reindex(dates).ffill()  # fehlende Tage vorwärts füllen
    date_series = pd.Series(pd.to_datetime(dates), index=pd.to_datetime(dates))  # Hilfsserie mit Tagesabständen
    days_to_next = (date_series.shift(-1) - date_series).dt.days.fillna(1).astype(int)  # Abstand zum nächsten Handelstag

    factor = 1.0 + rf * (days_to_next / float(day_count))  # einfache Verzinsung pro Intervall
    factor.iloc[-1] = 1.0  # letzter Tag hat keinen Folgetag

    close = factor.cumprod().astype("float64")  # kumulative Verzinsung als Schlusskurs
    open_ = close.shift(1).fillna(1.0)  # Open = Close des Vortags, Startwert 1.0
    high = np.maximum(open_.values, close.values)  # Tageshoch: Max aus Open/Close
    low = np.minimum(open_.values, close.values)   # Tagestief: Min aus Open/Close

    df_cash = pd.DataFrame(
        {
            # Rohschema beibehalten
            "open": open_.values,                          # synthetischer Eröffnungskurs
            "high": high,                                  # Tageshoch
            "low": low,                                    # Tagestief
            "close": close.values,                         # Schlusskurs
            "adj_close": close.values,                     # identisch mangels Splits
            "volume": 0.0,                                # kein Handelsvolumen
            "dividends": 0.0,                             # keine Dividenden
            "stock_splits": 1.0,                          # keine Splits
            # Core-Features
            "daily_return_log": np.log(factor.values),    # log. Tagesrendite
            "average_dollar_volume_20": 0.0,              # Liquidität irrelevant
            "volatility_becker_parkinson": 0.0,           # Volatilität = 0
            "bid_ask_spread_corwin_schultz": 0.0,         # Spread = 0
            # TA-Features: nicht sinnvoll für CASH
            "simple_moving_average_20": np.nan,           # Indikatoren bleiben NaN
            "simple_moving_average_60": np.nan,
            "exponential_moving_average_12": np.nan,
            "exponential_moving_average_26": np.nan,
            "relative_strength_index_14": np.nan,
            "macd_line_12_26_9": np.nan,
            "macd_signal_12_26_9": np.nan,
            "macd_histogram_12_26_9": np.nan,
            "bollinger_middle_band_20_2.0": np.nan,
            "bollinger_upper_band_20_2.0": np.nan,
            "bollinger_lower_band_20_2.0": np.nan,
            "bollinger_bandwidth_20_2.0": np.nan,
            "commodity_channel_index_20": np.nan,
            "average_directional_index_14": np.nan,
            "positive_directional_index_14": np.nan,
            "negative_directional_index_14": np.nan,
            # Exec/Flag
            "execution_price_t_plus_1_open": open_.shift(-1).values,  # Ausführungspreis nächster Tag
            "is_cash": 1,                                            # Kennzeichnung als CASH
        },
        index=dates,
    )
    df_cash["asset"] = symbol  # Spalte für Assetnamen hinzufügen
    return df_cash.reset_index().set_index(["date", "asset"])  # MultiIndex herstellen


def build_clean_data(
    prices: pd.DataFrame,
    risk_free_annual: pd.Series,
    out_path: Optional[str] = None,
    cash_symbol: str = "CASH",
    cs_sample_length: int = 1,   # Corwin–Schultz: Spanne (typisch 1–2)
 ) -> pd.DataFrame:
    """Feature-Panel mit technischen Kennzahlen und CASH-Asset erzeugen.

    Parameters
    ----------
    prices : pd.DataFrame
        Panel ``(date, asset)`` mit Rohpreisen.
    risk_free_annual : pd.Series
        Jahreszins pro Tag (dezimal, bereits auf Sessions ausgerichtet).
    out_path : str | None
        Optionaler Speicherpfad.
    cash_symbol : str
        Tickersymbol für das synthetische Cash.
    cs_sample_length : int
        Fenster für Corwin–Schultz-Spread-Schätzung.

    Returns
    -------
    pd.DataFrame
        Vollständiges Feature-Panel inklusive CASH.
    """
    # Input-Checks
    if not isinstance(prices.index, pd.MultiIndex) or prices.index.names != ["date", "asset"]:
        raise ValueError("prices muss MultiIndex mit Indexnamen ['date','asset'] besitzen.")
    if cash_symbol in prices.index.get_level_values("asset"):
        raise ValueError(f"Input darf {cash_symbol} noch nicht enthalten.")

    prices = prices.sort_index()  # sicherstellen, dass Daten zeitlich sortiert sind
    frames = []  # Sammelliste für Asset-DataFrames

    # --- Nicht-CASH Assets ---
    for asset, df_asset in prices.groupby(level="asset", sort=False):  # iteriere je Asset
        px = df_asset.droplevel("asset").sort_index()  # reine Ein-Asset-Serie

        # Core-Features
        daily_ret = returns(px["close"], kind="log")  # logarithmische Renditen
        adv20 = average_dollar_volume(px["close"], px["volume"], window=20)  # Liquidität

        beta = corwin_schultz_beta(px["high"], px["low"], sample_length=cs_sample_length)  # Spread-Proxies
        gamma = corwin_schultz_gamma(px["high"], px["low"])
        sigma_bp = becker_parkinson_sigma(beta, gamma)  # Volatilität aus High/Low

        alpha = corwin_schultz_alpha(beta, gamma)
        spread_cs = corwin_schultz_spread(alpha)  # Bid-Ask-Spread-Schätzung

        # TA-Features
        sma20 = simple_moving_average(px["close"], 20)  # kurzfristiger Trend
        sma60 = simple_moving_average(px["close"], 60)  # längerfristiger Trend
        ema12 = exponential_moving_average(px["close"], 12)  # schnell reagierend
        ema26 = exponential_moving_average(px["close"], 26)  # träge EMA
        rsi14 = relative_strength_index(px["close"], 14)  # Momentummaß
        macd_line, macd_signal, macd_hist = moving_average_convergence_divergence(px["close"], 12, 26, 9)
        boll_mid, boll_up, boll_lo, boll_bw = bollinger(px["close"], 20, 2.0)  # Bollinger-Bänder
        cci20 = commodity_channel_index(px["high"], px["low"], px["close"], 20)
        adx_df = average_directional_index(px["high"], px["low"], px["close"], 14)

        exec_ref = px["open"].shift(-1)  # Preis für t+1-Ausführung

        features = pd.DataFrame(
            {
                # Rohschema
                "open": px["open"],  # Eröffnungskurs
                "high": px["high"],  # Tageshoch
                "low": px["low"],    # Tagestief
                "close": px["close"],  # Schlusskurs
                "adj_close": px["adj_close"],  # bereinigter Kurs
                "volume": px["volume"].astype("float64"),  # Handelsvolumen
                "dividends": px["dividends"],  # ausgezahlte Dividenden
                "stock_splits": px["stock_splits"],  # Splitfaktor

                # Core
                "daily_return_log": daily_ret,  # log Rendite
                "average_dollar_volume_20": adv20,  # ADV20
                "volatility_becker_parkinson": sigma_bp,  # Volatilitätsmaß
                "bid_ask_spread_corwin_schultz": spread_cs,  # Spread-Schätzung

                # Technische Indikatoren
                "simple_moving_average_20": sma20,
                "simple_moving_average_60": sma60,
                "exponential_moving_average_12": ema12,
                "exponential_moving_average_26": ema26,
                "relative_strength_index_14": rsi14,
                "macd_line_12_26_9": macd_line,
                "macd_signal_12_26_9": macd_signal,
                "macd_histogram_12_26_9": macd_hist,
                "bollinger_middle_band_20_2.0": boll_mid,
                "bollinger_upper_band_20_2.0": boll_up,
                "bollinger_lower_band_20_2.0": boll_lo,
                "bollinger_bandwidth_20_2.0": boll_bw,
                "commodity_channel_index_20": cci20,
                "average_directional_index_14": adx_df["adx_14"],
                "positive_directional_index_14": adx_df["plus_di_14"],
                "negative_directional_index_14": adx_df["minus_di_14"],

                # Exec/Flag
                "execution_price_t_plus_1_open": exec_ref,  # für Simulation t+1
                "is_cash": 0,
            },
            index=px.index,
        )
        features["asset"] = asset  # Asset-Kennung hinzufügen
        frames.append(features.reset_index().set_index(["date", "asset"]))  # wieder MultiIndex

    # --- CASH Asset ---
    dates = prices.index.get_level_values(0).unique().sort_values()  # verfügbare Handelstage
    cash_df = _build_cash_asset(dates, risk_free_annual, day_count=360, symbol=cash_symbol)  # Kunst-Asset
    frames.append(cash_df)

    # --- Zusammenführen, Finalisieren ---
    panel = pd.concat(frames).sort_index()  # alles zusammenführen
    panel = panel[~panel.index.duplicated(keep="last")]  # doppelte Zeilen entfernen
    panel = _downcast_feature_dtypes(panel)  # Datentypen optimieren

    # Optional speichern
    if out_path:
        save_parquet(panel, out_path)  # persistieren

    return panel  # Feature-Panel zurückgeben


def write_clean_manifest(
    spec: dict,
    interim_path: str | Path,
    macro_path: str | Path,
    out_path: str | Path = "data/clean/features_v1.parquet",
    manifest_path: str | Path = "data/clean/_manifest.json",
) -> None:
    """Metadaten zur CLEAN-Stufe als Manifest ablegen.

    Parameters
    ----------
    spec : dict
        Verwendete Konfiguration.
    interim_path, macro_path : Path | str
        Eingangsdatensätze.
    out_path : Path | str
        Speicherort des Feature-Panels.
    manifest_path : Path | str
        Zielpfad für Manifest-Datei.
    """
    payload = {
        "stage": "clean",  # Pipeline-Stufe
        "dataset_id": spec.get("feature_version", "v1"),  # Versionierung
        "created_at": pd.Timestamp.utcnow().isoformat(),  # Zeitstempel
        "git_commit": current_commit_short(),  # Referenz aufs Repo
        "calendar": spec.get("align", {}).get("calendar", "XNYS"),  # verwendeter Kalender
        "spec": {
            "feature_version": spec.get("feature_version", "v1"),
            "windows": spec.get("windows", {}),
            "cs": spec.get("cs", {}),
            "risk_free": spec.get("risk_free", {}),
            "cash": spec.get("cash", {"symbol": "CASH"}),
        },
        "inputs": [file_summary(str(interim_path)), file_summary(str(macro_path))],  # Quellen
        "outputs": [file_summary(str(out_path))],  # erzeugte Dateien
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_manifest(payload, str(manifest_path))  # JSON auf Platte schreiben
