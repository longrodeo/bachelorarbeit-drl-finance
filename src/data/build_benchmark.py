# orchestrate_benchmark.py
# Zweck: ACWI (RAW) laden, dann ACWI + BTCUSD + GLD zu INTERIM-Panel bauen.

from datetime import date
import os

# 1) Projekt-Helfer
from src.utils.paths import INTERIM_PANEL  # Zielpfad für INTERIM
# 2) Loader für RAW (Tiingo)
from load_raw import download_raw_prices  # lädt und speichert RAW-Parquets :contentReference[oaicite:0]{index=0}
# 3) INTERIM-Builder (align + Kalender intern)
from build_interim import build_interim_prices  # baut (date, asset)-Panel & speichert nach INTERIM_PANEL :contentReference[oaicite:1]{index=1}

# ---------------------------------------------
# BENCHMARK-DEFINITION
ASSETS = ["ACWI", "BTC-USD", "IAU"]   # Gold-ETF ggf. auf "IAU" ändern
CRYPTO = {"BTC-USD"}                  # für korrektes Downsampling auf Handelstage

# ZEITFENSTER
START = "2015-01-01"
END   = "2024-12-31"     # oder fest setzen

# OPTIONAL: TIINGO_API_KEY muss für ACWI vorhanden sein
# os.environ["TIINGO_API_KEY"] = "<DEIN_KEY>"  # falls nicht in der Umgebung gesetzt

# ---------------------------------------------
# Schritt 1: RAW nur für fehlende/noch nicht geladene Assets ziehen (hier: ACWI)
# BTC & GLD sind laut Vorgabe bereits vorhanden; das ist idempotent, du kannst die Liste anpassen.
RAW_TO_FETCH = ["ACWI"]

print(f"[INFO] Lade RAW für: {RAW_TO_FETCH}")
_ = download_raw_prices(RAW_TO_FETCH, START, END)  # schreibt Parquets unter RAW/… :contentReference[oaicite:2]{index=2}

# Schritt 2: INTERIM-Panel bauen & speichern
# - Harmonisiert Spalten (open, high, low, close, adj_close, volume, dividends, stock_splits)
# - Align auf NYSE-Handelstage; Krypto wird auf Handelstage resampled (last)
# - Speichert nach src.utils.paths.INTERIM_PANEL
print(f"[INFO] Baue INTERIM-Panel für: {ASSETS}")
panel = build_interim_prices(
    assets=ASSETS,
    start=START,
    end=END,
    spec={
        "fields": ["open", "high", "low", "close", "adj_open", "adj_close", "volume", "dividends", "stock_splits"],
        "require_base_fields": True
    },
    crypto_assets=CRYPTO,
    save=True
)  # nutzt intern Handelskalender & Align-Helper :contentReference[oaicite:3]{index=3}

print(f"[OK] INTERIM geschrieben: {INTERIM_PANEL}")
print(panel.head())
