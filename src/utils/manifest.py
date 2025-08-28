# ---------------------------------------------------------------------------
# Datei: src/utils/manifest.py
# Zweck: Erzeugung von Manifest-Dateien mit Metadaten zu Datenbeständen
#   (z. B. Parquet-Dateien) inklusive Prüfsummen und Umgebungshinweisen.
# Hauptfunktionen: ``sha256_file``, ``file_summary``, ``current_commit_short``
#   und ``write_manifest``.
# Abhängigkeiten: Standardbibliotheken ``json``, ``hashlib`` usw. sowie ``pandas``
#   für Parquet-Inspektion.
# Edge Cases: fehlende Parquet-Unterstützung oder nicht-git-Repositories.
# ---------------------------------------------------------------------------
import json, hashlib, os, sys, platform, subprocess  # IO, Hashing und Systeminfo
import pandas as pd  # Lesen von Parquet-Dateien

def sha256_file(path: str, chunk_size: int = 1<<20) -> str:
    """Berechne SHA256-Prüfsumme einer Datei stückweise."""
    h = hashlib.sha256()  # Hashobjekt initialisieren
    with open(path, "rb") as f:  # Binärdatei öffnen
        while chunk := f.read(chunk_size):  # in Blöcken lesen (Speicherschonung)
            h.update(chunk)  # Hash laufend aktualisieren
    return h.hexdigest()  # finale hexadezimale Prüfsumme

def file_summary(path: str) -> dict:
    """Erzeuge Kurzbeschreibung einer Datei (insb. Parquet)."""
    # Für Parquet: Zeilen/Spalten/Datumsscope grob ermitteln
    try:  # Parquet lesen – kann bei Nicht-Parquet-Dateien fehlschlagen
        df = pd.read_parquet(path)  # DataFrame laden
        out = {  # Basis-Metadaten sammeln
            "path": path,
            "sha256": sha256_file(path),
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
        }
        if "date" in getattr(df.index, "names", []) or "date" in df.columns:
            d = df.reset_index()  # Index in Spalten überführen
            dates = d["date"] if "date" in d else d.set_index(df.index.names)["date"]
            out["date_min"] = str(pd.to_datetime(dates.min()).date())  # frühestes Datum
            out["date_max"] = str(pd.to_datetime(dates.max()).date())  # spätestes Datum
        return out  # vollständiges Summary zurückgeben
    except Exception:  # falls Lesen misslingt, nur Pfad & Hash ausgeben
        return {"path": path, "sha256": sha256_file(path)}

def current_commit_short() -> str:
    """Hole kurze Git-Commit-ID des aktuellen Arbeitsverzeichnisses."""
    try:  # git-Befehl kann scheitern (kein Repo)
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # Fallback, falls kein Git vorhanden
        return "unknown"

def write_manifest(payload: dict, out_path: str):
    """Speichere Manifest-Dictionary als JSON-Datei."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # Ausgabeverzeichnis sicherstellen
    with open(out_path, "w", encoding="utf-8") as f:  # Datei im Textmodus öffnen
        json.dump(payload, f, ensure_ascii=False, indent=2)  # schön formatiert schreiben
