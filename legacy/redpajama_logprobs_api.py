"""
redpajama_logprobs_api.py (ctx‑aware)
-----------------------------------
Stream a Parquet‑backed corpus through a **running vLLM OpenAI server** and
store per‑token log‑probabilities.  The script automatically splits any prompt
that exceeds the model’s context limit using a sliding‑window stride loop, so
requests never violate the server’s `max_context_length`.

Prerequisites
-------------
Start the server **without** chunked prefill (incompatible with sliding window)
and **with** an explicit sliding‑window size smaller than the model limit, e.g.:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --port 8000 \
  --no-enable-prefix-caching
```

Usage
-----
```bash
python redpajama_logprobs_api.py \
  --model Qwen/Qwen3-0.6B \
  --data-dir data/redpajama-1b \
  --file-pattern "0.parquet" \
  --output output/results.parquet \
  --batch-size 4 \
  --server-url http://localhost:8000/v1 \
  --max-context 40960 \
  --stride 2048 \
  --limit-docs 100
```

The `--max-context` / `--stride` pair must be ≤ the values the server was
started with.
"""
import argparse
import glob
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
# from openai import OpenAI
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_parquet_files(data_dir: str, pattern: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No Parquet files matching {pattern} under {data_dir}")
    return files


def stream_prompts(files: list[str], text_column: str = "text"):
    ds = load_dataset("parquet", data_files=files, streaming=True)["train"]
    for row in ds:
        prompt = row[text_column]
        if prompt:
            yield prompt


def evaluate_logprobs(server_url: str, model: str, prompt: str):
    """Return OpenAI response with prompt log‑probs enabled (no generation)."""

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    }

    headers = {
        "Content-Type": "application/json",
    }
    response = requests.post(f"{server_url}/v1/completions", json=payload, headers=headers)
    return response.json()


def extract_prompt_logprobs_and_decoded_tokens(choice):
    prompt_logprobs = choice.get("prompt_logprobs", None)
    if prompt_logprobs is None:
        return [], []
    try:
        logprobs, decoded_tokens = [], []
        for entry in prompt_logprobs:
            if entry is None:
                continue
            for token in entry:
                logprob = entry[token]["logprob"]
                decoded_token = entry[token]["decoded_token"]
                logprobs.append(logprob)
                decoded_tokens.append(decoded_token)
        return logprobs, decoded_tokens
    except AttributeError:
        return [], []


def sliding_window_logprobs(
    server_url: str,
    model: str,
    tokenizer,
    texts: list[str],
    ctx: int,
    stride: int,
):
    """
    Compute log‑probs for *text* without ever sending >ctx tokens.
    ctx: max context length
    stride: fresh tokens returned per window
    """
    toks = tokenizer.encode(texts, add_special_tokens=False)
    n = len(toks)
    logps: list[float] = []
    decoded_tokens: list[str] = []

    for cur in range(0, n, stride):
        beg = max(0, cur - (ctx - stride))
        window_toks = toks[beg : cur + stride]
        window_text = tokenizer.decode(window_toks)

        resp = evaluate_logprobs(server_url, model, window_text)
        choice = resp["choices"][0]
        lp, decoded_tokens = extract_prompt_logprobs_and_decoded_tokens(choice)
        logps.extend(lp)
        decoded_tokens.extend(decoded_tokens)

    return logps[:n], decoded_tokens[:n]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    parquet_files = find_parquet_files(args.data_dir, args.file_pattern)
    print(f"→ Found {len(parquet_files)} parquet files under {args.data_dir}")

    server_url = args.server_url
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    schema = pa.schema([
        pa.field("doc_id", pa.int64()),
        pa.field("prompt_logprobs", pa.list_(pa.float32())),
        pa.field("decoded_tokens", pa.list_(pa.string())),
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

            results = []
            for prompt in batch_prompts:
                lp, decoded_tokens = sliding_window_logprobs(
                    server_url,
                    args.model,
                    tokenizer,
                    prompt,
                    ctx=args.max_context,
                    stride=args.stride,
                )
                results.append({"doc_id": doc_id, "prompt_logprobs": lp, "decoded_tokens": decoded_tokens})
                doc_id += 1
                total_tokens += len(lp)

            writer.write_table(pa.Table.from_pylist(results, schema=schema))
            pbar.update(len(results))
            if args.limit_docs and doc_id >= args.limit_docs:
                break

        pbar.close()

    dt = time.time() - t0
    print(
        f"Processed {doc_id} documents | {total_tokens/1e6:.2f}M tokens | "
        f"{total_tokens/dt:,.0f} tok/s",
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="Extract token log‑probs via a vLLM OpenAI server with context‑aware splitting")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--file-pattern", default="*.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--server-url", default="http://0.0.0.0:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-context", type=int, default=-1, help="Max tokens allowed by the server / model; Default to model max context length")
    parser.add_argument("--stride", type=int, default=2048, help="Fresh tokens returned per window")
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
