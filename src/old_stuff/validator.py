# src/utils/validator.py
from __future__ import annotations

"""Validatoren für YAML-Konfigurationen.

- validate_assets prüft die Asset-Listen (equities/crypto).
- validate_spec prüft die Datenspezifikation.
- validate_all kombiniert beide Checks.

Bei Fehlern wird eine ValueError mit einer klaren Meldung ausgelöst.
"""

from pathlib import Path
from typing import Iterable
import argparse
import yaml


def _is_str_list(value: object) -> bool:
    """Hilfsfunktion: prüft, ob ``value`` eine nicht-leere Liste aus Strings ist."""
    return isinstance(value, list) and value and all(isinstance(x, str) for x in value)


def validate_assets(cfg: dict) -> None:
    """Validiert Assets-Konfigurationen.

    Erwartet die Schlüssel ``equities`` und ``crypto`` als nicht-leere Listen von
    Strings. Bei Verstößen wird eine ``ValueError`` geworfen.
    """
    equities = cfg.get("equities")
    crypto = cfg.get("crypto")

    if not _is_str_list(equities):
        raise ValueError("'equities' must be a non-empty list of strings")
    if not _is_str_list(crypto):
        raise ValueError("'crypto' must be a non-empty list of strings")


def validate_spec(spec: dict) -> None:
    """Validiert die Daten-Spezifikation.

    - ``fields`` muss alle erwarteten Spalten enthalten.
    - ``source`` muss entweder ``tiingo`` oder ``yahoo`` sein.
    """
    required_fields = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    ]

    fields = spec.get("fields")
    if not _is_str_list(fields):
        raise ValueError("'fields' must be a non-empty list of strings")

    missing = [f for f in required_fields if f not in fields]
    if missing:
        raise ValueError(f"'fields' missing required entries: {missing}")

    source = spec.get("source")
    if source not in {"tiingo", "yahoo"}:
        raise ValueError("'source' must be set to 'tiingo' or 'yahoo'")


def validate_all(cfg: dict, spec: dict) -> None:
    """Führt beide Validierungen aus."""
    validate_assets(cfg)
    validate_spec(spec)


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: Iterable[str] | None = None) -> None:
    """CLI-Einstiegspunkt."""
    parser = argparse.ArgumentParser(description="Validate YAML configs")
    parser.add_argument("--config", required=True, help="assets_*.yml file")
    parser.add_argument("--spec", required=True, help="data_spec.yml file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = _load_yaml(args.config)
    spec = _load_yaml(args.spec)

    validate_all(cfg, spec)
    print("[OK] Validation successful")


if __name__ == "__main__":  # pragma: no cover
    main()