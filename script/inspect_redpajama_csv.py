# read the result parquet file and print the first 10 rows

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def turn_csv_to_parquet(csv_path: str, parquet_path: str):
    df = pd.read_csv(csv_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

def inspect_length_distribution(df: pd.DataFrame, precentiles: list[float], text_column: str):
    # print the precentiles and means
    for percentile in precentiles:
        print(f"{percentile*100:.2f}%: {df[text_column].apply(len).quantile(percentile)}")

    print(f"Mean: {df[text_column].apply(len).mean()}")

if __name__ == "__main__":
    parquet_path = "/net/scratch2/listar2000/perplex_eval/data/redpajama-subset-50k.parquet"

    df = pq.read_table(parquet_path).to_pandas()

    inspect_length_distribution(df, [0.8, 0.85, 0.88], "text")