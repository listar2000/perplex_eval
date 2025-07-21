#!/usr/bin/env python3
"""
Chunk a Parquet file of texts using NLTK sentence segmentation.

* Input Parquet must contain columns:  id, text
* Each text is first split into sentences with nltk.sent_tokenize.
* Sentences are concatenated into ≈chunk_size-character blocks
  (never breaking a sentence in half).
* Results are written to:
      {output_dir}/chunked_texts_df.parquet
      {output_dir}/sample_chunks.txt   – first 20 chunks as a sanity check
"""

import argparse
import time
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

# One-time download (safe to call repeatedly)
nltk.download("punkt", quiet=True)


def chunk_text_nltk(text: str,
                    chunk_size: int = 500,
                    chunk_overlap: int = 0):
    """
    Split *text* into ~chunk_size-character chunks without cutting sentences.

    Parameters
    ----------
    text : str
        The raw document.
    chunk_size : int
        Target maximum characters per chunk.
    chunk_overlap : int
        Optional overlap (characters) between successive chunks.

    Returns
    -------
    list[str]
        List of chunk strings (order preserved).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Replace newlines with spaces before tokenizing
    cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()
    sentences = sent_tokenize(cleaned_text)

    chunks, current = [], ""

    for sent in sentences:
        sent = sent.strip()
        prospective_len = len(current) + (1 if current else 0) + len(sent)
        if prospective_len <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    # Add overlap if requested
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                overlap_text = chunks[i - 1][-chunk_overlap:]
                overlapped.append(overlap_text + chunk)
        chunks = overlapped

    return chunks


def main(input_parquet, output_dir,
         chunk_size=500, chunk_overlap=0):
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load input ──────────────────────────────────────────────────────────
    df = pd.read_parquet(input_parquet).drop_duplicates(subset="id")
    if not {"id", "text"}.issubset(df.columns):
        raise ValueError("Parquet must contain 'id' and 'text' columns.")

    # ─── Chunk all rows ──────────────────────────────────────────────────────
    rows = []
    for _, row in df.iterrows():
        chunks = chunk_text_nltk(row["text"],
                                 chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
        rows.extend({
            "text_id": row["id"],
            "chunk_id": i,
            "chunk_text": chunk
        } for i, chunk in enumerate(chunks))

    chunked_df = pd.DataFrame(rows)

    # ─── Save outputs ────────────────────────────────────────────────────────
    parquet_path = output_dir / "chunked_texts_df.parquet"
    chunked_df.to_parquet(parquet_path, index=False)

    sample_path = output_dir / "sample_chunks_2.txt"
    with sample_path.open("w", encoding="utf-8") as f:
        for _, r in chunked_df.head(20).iterrows():
            f.write(f"[text_id={r.text_id} | chunk_id={r.chunk_id}]\n")
            f.write(r.chunk_text.strip() + "\n\n")

    elapsed = time.time() - t0
    print(f"✅ Saved {len(chunked_df)} chunks to {parquet_path}")
    print(f"📄 Sample written to {sample_path}")
    print(f"⏱️  Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True,
                        help="Path to input Parquet file")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save outputs")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="Target characters per chunk (default: 500)")
    parser.add_argument("--chunk_overlap", type=int, default=0,
                        help="Optional characters of overlap (default: 0)")
    args = parser.parse_args()

    main(args.input_parquet, args.output_dir,
         chunk_size=args.chunk_size,
         chunk_overlap=args.chunk_overlap)