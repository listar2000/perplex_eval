import pandas as pd
import argparse
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
def main(input_parquet, output_dir):
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input parquet
    df = pd.read_parquet(input_parquet).drop_duplicates(subset='id')
    if not {'id', 'text'}.issubset(df.columns):
        raise ValueError("parquet must contain 'id' and 'text' columns.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        separators=["\n\n", "\n", "."]
    )

    chunked_rows = []
    for _, row in df.iterrows():
        text = row['text']
        text_id = row['id']
        chunks = splitter.split_text(text)
        for chunk_id, chunk in enumerate(chunks):
            chunked_rows.append({
                "text_id": text_id,
                "chunk_id": chunk_id,
                "chunk_text": chunk
            })

    chunked_df = pd.DataFrame(chunked_rows)

    # Save as Parquet
    parquet_path = output_dir / "chunked_texts_df.parquet"
    chunked_df.to_parquet(parquet_path, index=False)

    # Save a sample of the first 20 chunks
    sample_path = output_dir / "sample_chunks.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for _, row in chunked_df.head(20).iterrows():
            f.write(f"[text_id={row['text_id']} | chunk_id={row['chunk_id']}]\n")
            f.write(row['chunk_text'].strip() + "\n\n")

    print(f"✅ Saved {len(chunked_df)} chunks to {parquet_path}")
    print(f"📄 Sample written to {sample_path}")
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", type=str, required=True, help="Path to input parqruet")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    main(args.input_parquet, args.output_dir)