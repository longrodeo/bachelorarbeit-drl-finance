"""
Diagnose-Skript: zeigt Kopf und Meta-Informationen der erzeugten Parquet-Dateien
für INTERIM-, CLEAN- und Riskfree-Stufen. Kein echter Test, dient nur der
manuellen Sichtprüfung.
"""

from src.utils.paths import INTERIM_PANEL, CLEAN_PANEL, RISKFREE_FILE
from src.utils.parquet_io import load_parquet

print("=== INTERIM PANEL ===")  # Basisdaten nach Alignierung
interim = load_parquet(INTERIM_PANEL)
print("Shape:", interim.shape)
print("Index names:", interim.index.names)
print("Columns:", interim.columns.tolist()[:15])
print(interim.head(3))

print("\n=== CLEAN FEATURES ===")  # Feature-Panel
clean = load_parquet(CLEAN_PANEL)
print("Shape:", clean.shape)
print("Index names:", clean.index.names)
print("Columns:", clean.columns.tolist()[:15])
print(clean.head(3))

print("\n=== RISKFREE ===")  # Zinsserie
rf = load_parquet(RISKFREE_FILE)
print("Shape:", rf.shape)
print("Index name:", rf.index.name)
print(rf.head(3))
