import pandas as pd
from data.build_clean import build_clean_data
from utils.parquet_io import load_parquet, save_parquet

INTERIM = r"C:\Dev\Bachelorarbeit\data\interim\panel.parquet"      # anpassen
CLEAN   = r"C:\Dev\Bachelorarbeit\data\clean\features_v1.parquet"    # Ziel

prices = load_parquet(INTERIM)
build_clean_data(
    prices,
    out_path=CLEAN,
    cs_sample_length=2,   # dein neuer CS-Parameter
    verify=False          # beim Training meist False; bei Debug True
)
print("CLEAN neu geschrieben:", CLEAN)
