"""
Manuelles Skript zum Test verschiedener Parquet-Engines.
Schreibt/liest ein kleines DataFrame mit Default-, fastparquet- und pyarrow-Engine.
"""

# Standalone-Skript zur Verifikation der Parquet-Funktionalität ohne pytest.
# Dient als Smoke-Test für unterschiedliche Engines und zeigt etwaige
# Kompatibilitätsprobleme unmittelbar an.
# Nutzt pandas zum Schreiben/Lesen von Parquet-Dateien.
# Edge-Cases: fehlende Engine-Installationen werfen Exceptions.

import pandas as pd  # zentrale DataFrame-Bibliothek

print("Starte Parquet-Test...")  # Hinweis auf Start des Tests

# Kleines DataFrame
df = pd.DataFrame({
    "a": [1, 2, 3],  # erste Spalte mit Integers
    "b": [4.0, 5.0, 6.0]  # zweite Spalte mit Floats
})

print("DataFrame:")  # Ausgabe des DataFrames
print(df)

try:
    # Standard Engine
    print("\n>>> Schreibe mit default engine ...")
    df.to_parquet("tmp_default.parquet")  # nutzt pandas-Default (pyarrow oder fastparquet)
    out = pd.read_parquet("tmp_default.parquet")  # zurücklesen
    print("OK (default):", out.shape)

except Exception as e:
    print("Fehler mit default engine:", e)  # Fehlerausgabe

# Fastparquet
try:
    print("\n>>> Schreibe mit fastparquet ...")
    df.to_parquet("tmp_fast.parquet", engine="fastparquet")  # explizite Engine
    out = pd.read_parquet("tmp_fast.parquet", engine="fastparquet")
    print("OK (fastparquet):", out.shape)

except Exception as e:
    print("Fehler mit fastparquet:", e)

# PyArrow
try:
    print("\n>>> Schreibe mit pyarrow ...")
    df.to_parquet("tmp_arrow.parquet", engine="pyarrow")  # pyarrow als Engine
    out = pd.read_parquet("tmp_arrow.parquet", engine="pyarrow")
    print("OK (pyarrow):", out.shape)

except Exception as e:
    print("Fehler mit pyarrow:", e)

print("\nTest fertig.")  # Abschlussmeldung
