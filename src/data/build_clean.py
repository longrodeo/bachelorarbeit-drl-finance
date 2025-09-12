# ---------------------------------------------------------------------------
# Datei: src/data/build_clean.py
# Zweck: Kombiniert das INTERIM-Panel mit technischen Indikatoren. Ergebnis ist die CLEAN-Stufe.
# Hauptfunktionen: ``_build_cash_asset`` erzeugt das Kunst-Asset, ``build_clean_data``
#   berechnet Features und vereinigt alles, ``write_clean_manifest`` schreibt
#   Metadaten.
# Ein-/Ausgabe: MultiIndex-Panel ``(date, asset)`` → erweitertes Feature-Panel
#   sowie optionale Parquet/Manifest-Dateien.
# Abhängigkeiten: ``pandas``, ``numpy`` sowie eigene Feature-Module; Stolpersteine
#   sind falsche Datentypen.
# ---------------------------------------------------------------------------
"""
Erzeugt das finale Feature-Panel (CLEAN) inklusive synthetischem CASH-Asset.
Verknüpft Preisdaten mit technischen Indikatoren und speichert optional
Parquet-Dateien sowie ein Manifest. Abhängigkeiten reichen von NumPy/Pandas bis
zu eigenen Feature-Modulen. Typische Fehler: falsche Datentypen oder bereits
vorhandenes CASH-Asset im Input.
"""

# `from __future__`` erlaubt spätere Typreferenzen ohne String-Literale
from __future__ import annotations
# `Optional``-Alias für optionale Parameter bei Manifest/Output-Pfaden
from typing import Optional, Hashable
# Metainformationen über Python-Version usw. für Manifest
import platform  # Versionsinfo fürs Manifest
import pandas as pd  # Datenverarbeitung
from pathlib import Path  # Pfad-Manipulation


# Stabiler Parquet-Schreiber mit Engine-Fallbacks
from src.utils.parquet_io import save_parquet  # stabiler IO-Wrapper
# Manifest-Helfer für Prüfsummen und Commit-Referenzen
from src.utils.manifest import write_manifest, file_summary, current_commit_short  # Manifest-Helfer

# Feature-Funktionen aus euren Modulen
from src.features.basic_indicator import (
    returns,
    corwin_schultz_beta,
    corwin_schultz_gamma,
    corwin_schultz_alpha,
    becker_parkinson_sigma,     # Volaproxy
    corwin_schultz_spread_sanitized,      # finaler Spread
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
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")  # Float‑Features auf 32 Bit
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("int64")  # int64 für Mengen/Volumen
    return df  # DataFrame mit optimierten Datentypen zurückgeben

def _fmt_label(k: Hashable) -> str:
    return "/".join(map(str, k)) if isinstance(k, tuple) else str(k)


def build_clean_data(
    prices: pd.DataFrame,
    out_path: Optional[str] = None,
    cs_sample_length: int = 2, # Corwin–Schultz: Spanne (typisch 1–2)
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


    prices = prices.sort_index()  # sicherstellen, dass Daten zeitlich sortiert sind
    frames = []  # Sammelliste für Asset-DataFrames

    # --- Nicht-CASH Assets ---
    for asset, df_asset in prices.groupby(level="asset", sort=False):  # iteriere je Asset
        px = df_asset.droplevel("asset").sort_index()  # reine Ein-Asset-Serie

        # Core-Features
        daily_ret = returns(px["adj_close"], kind="log")  # logarithmische Renditen
        adv20 = average_dollar_volume(px["close"], px["volume"], window=20)  # Liquidität

        beta = corwin_schultz_beta(px["high"], px["low"], sample_length=cs_sample_length)  # Spread-Proxies
        gamma = corwin_schultz_gamma(px["high"], px["low"])
        sigma_bp = becker_parkinson_sigma(beta, gamma)  # Volatilität aus High/Low

        alpha = corwin_schultz_alpha(beta, gamma)

        CRYPTO_BASES = {"BTC-USD", "ETH-USD"}

        def _is_crypto_label(asset) -> bool:
            s = "/".join(map(str, asset)) if isinstance(asset, tuple) else str(asset)
            base = s.split("-")[0].upper()
            return base in CRYPTO_BASES or s.endswith("-USD")

        if _is_crypto_label(asset):
            # Konstante, konservative Crypto-Kosten (30 bp), optional glätten
            spread_cs = pd.Series(0.0030, index=px.index, dtype=float)
        else:
            # Sanitized CS
            spread_cs = corwin_schultz_spread_sanitized(alpha, roll=5, floor=1e-4)

        # Debug: wie viel alpha <= 0?  (einmalig ok)
        """neg = int((alpha <= 0).sum())
        tot = int(alpha.shape[0])
        print(f"[CS] {asset}: alpha<=0 = {neg}/{tot} ({neg / tot:.1%})")"""

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

        features = pd.DataFrame(
            {
                # Rohschema
                "open": px["open"],  # Eröffnungskurs
                "high": px["high"],  # Tageshoch
                "low": px["low"],    # Tagestief
                "close": px["close"],  # Schlusskurs
                "adj_open": px["adj_open"], # bereinigter Kurs
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
            },
            index=px.index,
        )
        features.index.name = "date"
        features = features.assign(asset=asset)  # Spalte hinzufügen
        features = features.set_index("asset", append=True)
        frames.append(features)  # MultiIndex: (date, asset)

    normed = []
    for f in frames:
        if isinstance(f, pd.Series):
            normed.append(f.to_frame(f.name or "value"))
        else:
            normed.append(f)

    # --- Zusammenführen, Finalisieren ---
    panel = pd.concat(normed).sort_index()  # alles zusammenführen
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
        },
        "inputs": [file_summary(str(interim_path)), file_summary(str(macro_path))],  # Quellen
        "outputs": [file_summary(str(out_path))],  # erzeugte Dateien
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_manifest(payload, str(manifest_path))  # JSON auf Platte schreiben
