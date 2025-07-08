"""
Evaluate the logprobs on the chunked-wiki dataset.

The script allows either data-level parallelism (DP) or tensor-level parallelism (TP).

Reference: https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html
"""
import os
import sys
import logging
from time import sleep
from multiprocessing import Process, Queue
import pandas as pd
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from vllm import LLM, SamplingParams
from vllm.sequence import Logprob
# from vllm.utils import get_open_port
import yaml


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/wiki/chunked_texts_df.parquet",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="chunk_text",
        help="Column name of the dataset to use as prompts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vllm_config.yaml",
        help="Path to YAML config file for vLLM arguments",
    )
    # output related arguments
    parser.add_argument(
        "--output-file", type=str, default="output/wiki.parquet", help="Path for the output Parquet file"
    )
    parser.add_argument("--debug-limit", type=int, default=0, help="Limit the number of prompts to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose vLLM output")
    return parser.parse_args()


def load_vllm_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_logprobs(raw_logprobs: list[Optional[dict[int, Logprob]]]) -> list[float]:
    logprobs = []
    # the first one is always None, so we skip it
    for logprob in raw_logprobs[1:]:
        if logprob is not None:
            logprobs.append(next(iter(logprob.values())).logprob)
    return logprobs

def configure_logging(verbose: bool = False):
    """Configure logging to control vLLM verbosity"""
    if not verbose:
        # Suppress vLLM's verbose output
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("vllm.engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.worker").setLevel(logging.WARNING)
        logging.getLogger("vllm.distributed").setLevel(logging.WARNING)
        # Also suppress other potentially verbose loggers
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Force stdout to flush immediately for real-time output
    sys.stdout.flush()

def get_data_slice(dataset: pd.DataFrame, dp_size: int, global_dp_rank: int):
    """Get the data slice for a specific DP rank"""
    floor = len(dataset) // dp_size
    remainder = len(dataset) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)
    
    beg_idx, end_idx = start(global_dp_rank), start(global_dp_rank + 1)
    return dataset.iloc[beg_idx:end_idx]

def process_dataset_slice(
    model: str,
    tp_size: int,
    vllm_config: dict,
    dataset_slice: pd.DataFrame,
    prompt_column: str,
    global_dp_rank: int,
    verbose: bool = False,
):
    """Process a slice of the dataset and return results"""
    configure_logging(verbose)
    
    try:
        # Set environment variables when DP=1
        # os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        # os.environ["VLLM_DP_RANK_LOCAL"] = str(global_dp_rank)
        # os.environ["VLLM_DP_SIZE"] = str(1)  # Single process
        # os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        # os.environ["VLLM_DP_MASTER_PORT"] = str(8000)

        prompt_dicts = dataset_slice.to_dict(orient="records")
        prompts = [prompt_dict[prompt_column] for prompt_dict in prompt_dicts]

        assert len(prompt_dicts) > 0, f"Dataset has no prompts"
        print(f"Single process: Will process {len(prompt_dicts)} prompts", flush=True)

        # Create a sampling params object.
        sampling_params = SamplingParams(
            temperature=0.0, prompt_logprobs=0, max_tokens=1
        )

        # Create an LLM.
        print(f"Initializing LLM...", flush=True)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            **vllm_config,
        )
        
        print(f"Single process: Generating outputs...", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"Single process: Processing {len(outputs)} outputs...", flush=True)
        results = []
        for i, output in enumerate(outputs):
            text_id, chunk_id = int(prompt_dicts[i]["text_id"]), int(prompt_dicts[i]["chunk_id"])
            logprobs = process_logprobs(output.prompt_logprobs)
            results.append({"text_id": text_id, "chunk_id": chunk_id, "logprobs": logprobs})

        print(f"Single process: Completed processing", flush=True)
    
        return results
        
    except Exception as e:
        print(f"Single process: ERROR - {str(e)}", flush=True)
        raise e

def main(
    model: str,
    dp_size: int,
    tp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    vllm_config: dict,
    dataset: pd.DataFrame,
    prompt_column: str,
    results_queue: Queue,
    verbose: bool = False,
):
    """Main function for multiprocessing workers (DP > 1)"""
    configure_logging(verbose)
    
    try:
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        # Get data slice for this rank
        dataset_slice = get_data_slice(dataset, dp_size, global_dp_rank)
        prompt_dicts = dataset_slice.to_dict(orient="records")
        prompts = [prompt_dict[prompt_column] for prompt_dict in prompt_dicts]

        assert len(prompt_dicts) > 0, f"DP rank {global_dp_rank} has no prompts"
        print(f"DP rank {global_dp_rank} processing {len(prompt_dicts)} prompts", flush=True)

        # Create a sampling params object.
        sampling_params = SamplingParams(
            temperature=0.0, prompt_logprobs=0, max_tokens=1
        )

        # Create an LLM.
        print(f"DP rank {global_dp_rank}: Initializing LLM...", flush=True)
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            **vllm_config,
        )
        
        print(f"DP rank {global_dp_rank}: Generating outputs...", flush=True)
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"DP rank {global_dp_rank}: Processing {len(outputs)} outputs...", flush=True)
        for i, output in enumerate(outputs):
            text_id, chunk_id = int(prompt_dicts[i]["text_id"]), int(prompt_dicts[i]["chunk_id"])
            logprobs = process_logprobs(output.prompt_logprobs)
            results_queue.put({"text_id": text_id, "chunk_id": chunk_id, "logprobs": logprobs})

        # Signal that this worker is done by putting None in the queue
        results_queue.put(None)
        print(f"DP rank {global_dp_rank}: Completed processing", flush=True)
        
        # Give engines time to pause their processing loops before exiting. 
        sleep(1)
        
    except Exception as e:
        print(f"DP rank {global_dp_rank}: ERROR - {str(e)}", flush=True)
        # Re-raise the exception to ensure the process exits with error code
        raise

def write_results_to_parquet(results_batch, parquet_writer, schema, output_filename):
    """Write a batch of results to the Parquet file"""
    if not results_batch:
        return
    
    # Convert the batch of dictionaries to a pyarrow Table
    table = pa.Table.from_pydict({
        'text_id': [r['text_id'] for r in results_batch],
        'chunk_id': [r['chunk_id'] for r in results_batch],
        'logprobs': [r['logprobs'] for r in results_batch]
    }, schema=schema)

    # Write the table to the Parquet file
    parquet_writer.write_table(table)
    print(f"Wrote a batch of {len(results_batch)} results to {output_filename}", flush=True)

def run_single_process(
    model: str,
    tp_size: int,
    vllm_config: dict,
    dataset: pd.DataFrame,
    prompt_column: str,
    output_filename: str,
    verbose: bool = False,
):
    """Run inference in single process mode (DP=1)"""
    print("Running in single process mode (DP=1)...", flush=True)
    
    # Define the schema for the Parquet file
    schema = pa.schema([
        pa.field('text_id', pa.int32()),
        pa.field('chunk_id', pa.int32()),
        pa.field('logprobs', pa.list_(pa.float32()))
    ])

    results_batch = []
    batch_size = 5000  # Write to file every 5000 results (higher for single process)

    try:
        # Open the Parquet writer
        parquet_writer = pq.ParquetWriter(output_filename, schema)
        print(f"Opened Parquet writer for {output_filename}", flush=True)

        # Process the entire dataset
        results = process_dataset_slice(
            model=model,
            tp_size=tp_size,
            vllm_config=vllm_config,
            dataset_slice=dataset,
            prompt_column=prompt_column,
            global_dp_rank=0,
            verbose=verbose,
        )
        
        # Write results in batches
        for result in results:
            results_batch.append(result)
            
            if len(results_batch) >= batch_size:
                write_results_to_parquet(results_batch, parquet_writer, schema, output_filename)
                results_batch = []
        
        # Write any remaining results
        if results_batch:
            write_results_to_parquet(results_batch, parquet_writer, schema, output_filename)
            
    finally:
        if parquet_writer:
            parquet_writer.close()
        print(f"Finished writing all results to {output_filename}.", flush=True)

def run_multi_process(
    model: str,
    dp_size: int,
    tp_size: int,
    vllm_config: dict,
    dataset: pd.DataFrame,
    prompt_column: str,
    output_filename: str,
    verbose: bool = False,
):
    """Run inference in multiprocessing mode (DP>1)"""
    print(f"Starting {dp_size} data parallel processes...", flush=True)
    
    dp_master_ip = "127.0.0.1"
    dp_master_port = 8000
    results_queue = Queue()
    
    # Define the schema for the Parquet file
    schema = pa.schema([
        pa.field('text_id', pa.int32()),
        pa.field('chunk_id', pa.int32()),
        pa.field('logprobs', pa.list_(pa.float32()))
    ])

    results_batch = []
    batch_size = 1000  # Write to file every 1000 results

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(dp_size)):
        proc = Process(
            target=main,
            args=(
                model,
                dp_size,
                tp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                vllm_config,
                dataset,
                prompt_column,
                results_queue,
                verbose,
            ),
        )
        proc.start()
        procs.append(proc)
        print(f"Started process {proc.pid} for DP rank {global_dp_rank}", flush=True)

    # Logic to collect results and stream to Parquet
    finished_workers = 0
    parquet_writer = None

    try:
        # Open the Parquet writer
        parquet_writer = pq.ParquetWriter(output_filename, schema)
        print(f"Opened Parquet writer for {output_filename}", flush=True)

        while finished_workers < len(procs):
            # Get a result from the queue
            result = results_queue.get()

            if result is None:
                # A worker has finished
                finished_workers += 1
                print(f"Worker {finished_workers}/{len(procs)} completed", flush=True)
            else:
                results_batch.append(result)
            
            # If batch is full or all workers are done and there are leftover results
            if len(results_batch) >= batch_size or (finished_workers == len(procs) and results_batch):
                write_results_to_parquet(results_batch, parquet_writer, schema, output_filename)
                # Clear the batch
                results_batch = []
    finally:
        if parquet_writer:
            parquet_writer.close()
        print(f"Finished writing all results to {output_filename}.", flush=True)

    # Wait for all processes to complete
    print("Waiting for all processes to complete...", flush=True)
    exit_code = 0
    for i, proc in enumerate(procs):
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} (DP rank {i}) that didn't stop within 5 minutes.", flush=True)
            proc.kill()
            exit_code = 1
        elif proc.exitcode != 0:
            print(f"Process {proc.pid} (DP rank {i}) exited with code {proc.exitcode}", flush=True)
            exit_code = proc.exitcode
        else:
            print(f"Process {proc.pid} (DP rank {i}) completed successfully", flush=True)

    if exit_code == 0:
        print("All processes completed successfully!", flush=True)
    else:
        print(f"Some processes failed with exit code {exit_code}", flush=True)

    return exit_code

if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    model = args.model
    output_filename = args.output_file

    # Load vLLM config from YAML
    vllm_config = load_vllm_config(args.config)

    # Load and prepare dataset
    wiki_dataset = pd.read_parquet(args.dataset)
    print(f"Total number of chunks/requests: {len(wiki_dataset)}")
    
    if args.debug_limit > 0:
        wiki_dataset = wiki_dataset.head(args.debug_limit)
        print(f"Debug mode: only processing {args.debug_limit} prompts")

    if dp_size == 1:
        run_single_process(
            model=model,
            tp_size=tp_size,
            vllm_config=vllm_config,
            dataset=wiki_dataset,
            prompt_column=args.prompt_column,
            output_filename=output_filename,
            verbose=args.verbose,
        )
    else:
        exit_code = run_multi_process(
            model=model,
            dp_size=dp_size,
            tp_size=tp_size,
            vllm_config=vllm_config,
            dataset=wiki_dataset,
            prompt_column=args.prompt_column,
            output_filename=output_filename,
            verbose=args.verbose,
        )
        exit(exit_code)
