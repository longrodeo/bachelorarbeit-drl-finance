import pandas as pd

print("Starte Parquet-Test...")

# Kleines DataFrame
df = pd.DataFrame({
    "a": [1, 2, 3],
    "b": [4.0, 5.0, 6.0]
})

print("DataFrame:")
print(df)

try:
    # Standard Engine
    print("\n>>> Schreibe mit default engine ...")
    df.to_parquet("tmp_default.parquet")
    out = pd.read_parquet("tmp_default.parquet")
    print("OK (default):", out.shape)

except Exception as e:
    print("Fehler mit default engine:", e)

# Fastparquet
try:
    print("\n>>> Schreibe mit fastparquet ...")
    df.to_parquet("tmp_fast.parquet", engine="fastparquet")
    out = pd.read_parquet("tmp_fast.parquet", engine="fastparquet")
    print("OK (fastparquet):", out.shape)

except Exception as e:
    print("Fehler mit fastparquet:", e)

# PyArrow
try:
    print("\n>>> Schreibe mit pyarrow ...")
    df.to_parquet("tmp_arrow.parquet", engine="pyarrow")
    out = pd.read_parquet("tmp_arrow.parquet", engine="pyarrow")
    print("OK (pyarrow):", out.shape)

except Exception as e:
    print("Fehler mit pyarrow:", e)

print("\nTest fertig.")
