import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parquet-file", type=str, default="output/wiki.parquet")
args = parser.parse_args()

df = pd.read_parquet(args.parquet_file)
# df = pd.read_parquet('output/wiki.parquet')

# Column names
print(df.columns)

# Number of unique texts
num_texts = df['text_id'].nunique()
print(f"Number of unique texts: {num_texts}")

# Chunks per text
chunks_per_text = df.groupby('text_id').size()
print(f"\nAverage number of chunks per text: {chunks_per_text.mean():.2f}")
print("\nTop 5 texts with the most chunks:")
print(chunks_per_text.nlargest(5))

# Chunk text length
# df['chunk_length'] = df['chunk_text'].str.len()
# print(f"\nAverage length of chunk_text: {df['chunk_length'].mean():.2f}")
# print("\nTop 5 longest chunk_texts (showing length):")
# print(df.nlargest(5, 'chunk_length')[['text_id', 'chunk_id', 'chunk_length']])
