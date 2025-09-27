
from src.data.build_clean import build_clean_data
from src.utils.parquet_io import load_parquet


df = load_parquet("C:/Dev/Bachelorarbeit/data/interim/benchmark.parquet")
out = "C:/Dev/Bachelorarbeit/data/clean/benchmark.parquet"

build_clean_data(prices=df, cs_sample_length=2, out_path=out)