#!/usr/bin/env python3
"""
Chunk a Parquet file of texts using NLTK sentence segmentation.

Changes vs. previous version
────────────────────────────
1. Overlap is snapped to word boundaries (never splits words).
2. Instead of `new_text`, two indices `new_start`, `new_end` are stored so the
   caller can slice `chunk_text` directly.
"""

import argparse, time, re
from pathlib import Path

import nltk, pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download("punkt", download_dir="./cache", quiet=True)

_WORD_RE = re.compile(r"\w")


def _tail_with_word_boundary(text: str, min_len: int) -> str:
    """
    Return a suffix of *text* of ≥*min_len* characters whose first char is a
    word boundary (either start-of-text or preceded by non-word).
    """
    if min_len >= len(text):
        return text
    start = len(text) - min_len
    # Walk left while we are inside a word
    while start > 0 and _WORD_RE.match(text[start]) and _WORD_RE.match(text[start - 1]):
        start -= 1
    return text[start:]


def chunk_text_nltk(text: str, chunk_size: int = 500):
    """Return a list of base (non-overlapped) chunks."""
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    sentences = sent_tokenize(cleaned)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if len(cur) + (1 if cur else 0) + len(s) <= chunk_size:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def main(input_parquet, output_dir, chunk_size=500, chunk_overlap=0):
    t0 = time.time()
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_parquet).drop_duplicates(subset="id")
    if not {"id", "text"}.issubset(df.columns):
        raise ValueError("Parquet must contain 'id' and 'text' columns")

    rows = []
    for _, row in df.iterrows():
        bases = chunk_text_nltk(row["text"], chunk_size)
        for i, base in enumerate(bases):
            if i == 0 or chunk_overlap == 0:
                chunk_text = base
                new_start = 0
            else:
                overlap = _tail_with_word_boundary(bases[i - 1], chunk_overlap)
                chunk_text = overlap + base
                new_start = len(overlap)
            new_end = len(chunk_text)

            rows.append({
                "text_id":  row["id"],
                "chunk_id": i,
                "chunk_text": chunk_text,
                "new_start": new_start,
                "new_end":   new_end
            })

    chunked_df = pd.DataFrame(rows)
    parquet_path = out_dir / "chunked_texts_df.parquet"
    chunked_df.to_parquet(parquet_path, index=False)

    sample = out_dir / "sample_chunks.txt"
    with sample.open("w", encoding="utf-8") as f:
        for _, r in chunked_df.head(20).iterrows():
            f.write(f"[id={r.text_id} | chunk={r.chunk_id}]\n")
            f.write("⋯full⋯ " + r.chunk_text + "\n")
            f.write("⋯new⋯  " + r.chunk_text[r.new_start:r.new_end] + "\n\n")

    print(f"✅ {len(chunked_df)} rows written → {parquet_path}")
    print(f"📄 Sample → {sample}")
    print(f"⏱️  Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_parquet", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--chunk_overlap", type=int, default=100)
    a = p.parse_args()
    main(a.input_parquet, a.output_dir,
         chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap)
