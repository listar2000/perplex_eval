import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="output/wiki.parquet")
args = parser.parse_args()

df = pd.read_parquet(args.file)
# df = pd.read_parquet('output/wiki.parquet')

# Number of rows: 39961600
print(f"Number of rows: {len(df)}")

# Column names
print("Columns:")
print(df.columns)

# Sample 10 random rows and print them
print("\nSample rows:")
print(df.sample(10))

# get the dtype for each column
print("\nDtypes:")
print(df.dtypes)
