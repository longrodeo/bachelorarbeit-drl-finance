# Bachelorarbeit – DRL Backtest (Reward-Design)

Evaluationsstarke Backtests für Deep Reinforcement Learning im Finanzbereich.  
Fokus: **Reward-Design** unter **strengen, reproduzierbaren** Tests (Daten-/State-Leak-frei, T+1-Ausführung, Kosten/Spreads).

---

## 🚀 Projektüberblick

- **Single Source of Truth (Daten):** Tiingo (Equities & Crypto), konsequent **snake_case** Felder.
- **Zeit & Kalender:** Daten auf **NYSE-Handelstage** aligned; Krypto optional `ffill`.
- **Features:** Returns, ADV, Parkinson-Volatilität, **Corwin–Schultz** (σ, Spread), Exec-Referenz *T+1*.
- **Persistenz:** Parquet via **robusten Wrapper** (`fastparquet` bevorzugt, `pyarrow` Fallback).
- **Tests:** Schnelle Unit-Tests + langsame End-to-End-Smoke-Tests.

---

## 📦 Struktur

```
.
├─ config/
│  ├─ assets_example.yml
│  ├─ assets_ucits_compact.yml
│  └─ data_spec.yml
├─ src/
│  ├─ data/
│  │  └─ loader.py
│  └─ utils/
│     ├─ parquet_io.py
│     └─ validator.py
├─ tests/
│  ├─ test_loader.py
│  └─ test_loader_smoke.py
├─ data/
│  └─ raw/
│     └─ <name>/
│        ├─ panel.parquet
│        ├─ <ASSET>.parquet
│        └─ manifest.json   (optional, s.u.)
└─ pytest.ini
```

---

## ⚙️ Konfiguration

### `config/data_spec.yml` (Ausschnitt)
```yaml
fields: ["open","high","low","close","adj_close","volume","dividends","stock_splits"]
return_type: log
align:
  ffill_crypto: true
windows:
  adv: 20
  sigma_hl: 20
  spread_cs: 2
```

### Asset-Listen (Beispiel)
```yaml
# config/assets_example.yml
name: example
start: 2018-01-01
end: 2024-12-31
equities: ["SPY", "IEFA"]
crypto:   ["BTC-USD"]
```

```yaml
# config/assets_ucits_compact.yml
name: ucits_compact
start: 2018-01-01
end: 2024-12-31
equities: ["SPY", "IEFA", "IEMG", "IAU"]
crypto:   ["BTC-USD", "ETH-USD"]
```

> **Hinweis:** Ticker **immer in Anführungszeichen** (insb. `BTC-USD`), sonst kann YAML fehlinterpretieren.

---

## 🧠 Daten- & Feature-Konventionen

- **Index:** MultiIndex `["date","asset"]`, keine Duplikate.
- **Pflichtspalten (raw):** `open, high, low, close, adj_close, volume, dividends, stock_splits`.
- **Derived:**
  - `return_raw` – log/pct Change von `close`.
  - `adv` – Average Dollar Volume (`close * volume`, roll. Mittel).
  - `sigma_hl` – Parkinson-Volatilität (HL-basiert, rolling).
  - `sigma_cs`, `spread_cs` – **Corwin–Schultz** (Beta/Gamma → Alpha → Sigma/Spread).
  - `exec_ref_tplus1` – Ausführungsreferenz **Open_{t+1}** (für T+1-Backtests).

---

## 🧪 Tests

**PyTest-Marker deklarieren** (`pytest.ini`):
```ini
[pytest]
markers =
  slow: end-to-end loader & IO
```

**Schnelltests (CI-like):**
```bash
pytest -q tests/test_loader.py
```

**End-to-End Smoke:**
```bash
pytest -q -m slow tests/test_loader_smoke.py
```

**Validator (Schemas checken):**
```bash
python -m src.utils.validator --config config/assets_example.yml --spec config/data_spec.yml
```

---

## ⬇️ Daten herunterladen (Bulk)

```bash
# TIINGO_API_KEY muss als Umgebungsvariable gesetzt sein
python -m src.data.loader --config config/assets_ucits_compact.yml --spec config/data_spec.yml
```

**Optionales Logging:**
```bash
mkdir -p logs
python -m src.data.loader --config config/assets_ucits_compact.yml --spec config/data_spec.yml >> logs/bulk_download.log 2>&1
```

**Ergebnis prüfen (Quick-Check):**
```python
from src.utils.parquet_io import load_parquet
df = load_parquet("data/raw/ucits_compact/panel.parquet")
print(df.shape, df.index.names, df.columns[:10])
```

---

## 🧾 Reproduzierbarkeit (Manifest, optional)

Empfohlen: im `save_panel()` ein `manifest.json` schreiben (Assets, Shape, Spalten, Timestamp).  
So lässt sich jeder Run in der BA exakt referenzieren.

**Beispiel-Payload:**
```json
{
  "name": "ucits_compact",
  "assets": ["SPY","IEFA","IEMG","IAU","BTC-USD","ETH-USD"],
  "shape": [123456, 14],
  "columns": ["open","high","low","close","adj_close","volume","dividends","stock_splits","return_raw","adv","sigma_hl","sigma_cs","spread_cs","exec_ref_tplus1"],
  "created": "2025-08-21T10:15:30"
}
```

---

## 🧩 Design-Entscheidungen

- **Provider fix:** Tiingo only (Yahoo entfernt) → weniger Variablen, bessere Replizierbarkeit.
- **Spalten-Schema:** kanonisches **snake_case** end-to-end.
- **Parquet-IO:** zentral über `save_parquet`/`load_parquet` (bevorzugt `fastparquet`, Fallback `pyarrow`), um Windows-Crashes zu vermeiden.
- **Kalender:** NYSE-Sessions statt „BusinessDays“.
- **T+1-Ausführung:** Signale an Tag _t_, Ausführung (Referenzpreis) an **Open_{t+1}**.

---

## 🛠️ Umgebung (Kurz)

- Python 3.10 (Conda-Env empfohlen).
- Pakete: `pandas`, `numpy`, `requests`, `exchange_calendars` (optional), `fastparquet` (oder `pyarrow`).
- Setze `TIINGO_API_KEY` als Umgebungsvariable.

---

## 🐞 Troubleshooting

- **Leeres Panel / keine Dateien:**
  - Prüfe `name`, `start/end`, **flache Listen** in der Config, korrekte Ticker.
  - Sieh ins Log (`logs/bulk_download.log`) – Rate-Limit/HTTP-Fehler.
- **Windows Crash (0xc000001d) bei Parquet:**
  - Nutze die Wrapper (`parquet_io.py`), die `fastparquet` priorisieren.
- **Spalten fehlen:**
  - `data_spec.yml` muss exakt die snake_case-Felder definieren (siehe oben).
- **Index-Fehler / Duplikate:**
  - Kalender-Align + `assert_no_dupes` prüft und stoppt mit klarer Meldung.

---

## 📍 Nächste Schritte (Ticket 3)

- Kosten/Slippage ins Exec-Layer integrieren (Spread-aware T+1-Fill).
- Backtest-Engine hooken (z. B. vectorbt/own) mit T+1-Signals, Positions-Bounds, Budget-Projection.
- Erste Baselines + Reward-Varianten evaluieren (Sharpe/Sortino/MaxDD, Walk-forward).

---

## 📄 Lizenz / Zitation

Dieses Repo ist Bestandteil einer Bachelorarbeit (2025/26).  
Bitte nur zu Forschungs-/Lehrzwecken verwenden und Tiingo-Nutzungsbedingungen beachten.
