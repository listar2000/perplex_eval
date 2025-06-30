"""
redpajama_logprobs_api.py
------------------------
Stream a Parquet‑backed corpus (e.g. RedPajama‑1B) through a **running vLLM
OpenAI server** and store per‑token log‑probabilities in a Parquet file.

The server must be launched beforehand, for example:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-0.6B \
    --tensor-parallel-size 2 \
    --enable-chunked-prefill \
    --long-prefill-token-threshold 2048 \
    --max-num-partial-prefills 2 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.9
```

The script keeps the original data‑loading / saving logic but replaces the
in‑process `LLM` object with the **OpenAI Python client**, letting the server
handle chunked prefill and KV management.

Example
-------
```bash
python run_redpajama_1b_inference.py \
  --model Qwen/Qwen3-0.6B \
  --data-dir data/redpajama-1b \
  --file-pattern "0.parquet" \
  --output output/results.parquet \
  --batch-size 8 \
  --server-url http://localhost:8000/v1 \
  --api-key EMPTY
```
"""

import argparse
import glob
import os
import time
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_parquet_files(data_dir: str, pattern: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No Parquet files matching {pattern} under {data_dir}")
    return files


def stream_prompts(files: List[str], text_column: str = "text", max_length: int = 10000):
    ds = load_dataset("parquet", data_files=files, streaming=True)["train"]
    for row in ds:
        prompt = row[text_column]
        if prompt and len(prompt) <= max_length:
            yield prompt


def extract_prompt_logprobs(choice):
    """Return the per‑token log‑probs for the *prompt* part of a choice."""
    lp_container = getattr(choice, "logprobs", None)
    if lp_container is None:
        return []
    # vLLM returns an object with attribute lists; fall back to dict‑style.
    try:
        return list(lp_container.token_logprobs)
    except AttributeError:
        return list(lp_container["token_logprobs"])

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(args):
    parquet_files = find_parquet_files(args.data_dir, args.file_pattern)
    print(f"→ Found {len(parquet_files)} parquet files under {args.data_dir}")

    client = OpenAI(api_key=args.api_key, base_url=args.server_url)

    schema = pa.schema([
        pa.field("doc_id", pa.int64()),
        pa.field("logprobs", pa.list_(pa.float32())),
    ])

    doc_stream = stream_prompts(parquet_files)

    total_tokens = 0
    doc_id = 0
    t0 = time.time()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with pq.ParquetWriter(args.output, schema) as writer:
        pbar = tqdm(total=args.limit_docs or None, desc="documents", unit="doc")
        while True:
            batch_prompts = []
            try:
                for _ in range(args.batch_size):
                    batch_prompts.append(next(doc_stream))
            except StopIteration:
                pass
            if not batch_prompts:
                break

            # Call the vLLM OpenAI endpoint with *multiple* prompts at once.
            resp = client.completions.create(
                model=args.model,
                prompt=batch_prompts,
                max_tokens=1,
                temperature=0.0,
                logprobs=1,
            )

            results = []
            for choice in resp.choices:
                lp = extract_prompt_logprobs(choice)
                results.append({"doc_id": doc_id, "logprobs": lp})
                doc_id += 1
                total_tokens += len(lp)

            writer.write_table(pa.Table.from_pylist(results, schema=schema))
            pbar.update(len(results))
            if args.limit_docs and doc_id >= args.limit_docs:
                break

        pbar.close()

    dt = time.time() - t0
    print(
        f"Processed {doc_id} documents | {total_tokens/1e3:.2f}k tokens | "
        f"{total_tokens/dt:,.0f} tok/s"
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Extract token log‑probs via a vLLM OpenAI server")
    parser.add_argument("--model", required=True, help="Model name passed to the server")
    parser.add_argument("--data-dir", required=True, help="Directory holding Parquet shards")
    parser.add_argument("--file-pattern", default="*.parquet", help="Globbing pattern for shards")
    parser.add_argument("--output", required=True, help="Output Parquet file path")
    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per API call")
    parser.add_argument("--server-url", default="http://localhost:8000/v1", help="Base URL of the vLLM server")
    parser.add_argument("--api-key", default="EMPTY", help="Dummy key for local server")
    parser.add_argument("--limit-docs", type=int, default=None, help="Stop early after N documents (debug)")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
