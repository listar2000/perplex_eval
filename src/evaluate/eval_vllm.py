"""
Evaluate the "normalized" logprobs of a document with token-based filtering.

This script loads a dataset, filters texts based on token count (using the model's tokenizer),
and then evaluates them using vLLM to compute logprobs and perplexity scores.

A sample usage:
```bash
python -m src.evaluate.eval_vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tp_size 2 \
    --vllm_config_path vllm_config.yaml \
    --dataset_path data/redpajama-subset-50k.parquet \
    --save_folder output/ \
    --max_len 8192
```
"""
import logging
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from .vllm_utils import process_vllm_logprobs, set_all_logging_level, load_vllm_config

logger = logging.getLogger(__name__)


def vllm_process_documents(
    model: str,
    tp_size: int,
    vllm_config: dict,
    dataset: pd.DataFrame,
    prompt_column: str,
):
    """The core function for invoking vLLM to process documents."""
    
    try:
        prompts = dataset[prompt_column].tolist()

        assert len(prompts) > 0, logger.error(f"Dataset has no prompts")
        logger.info(f"Single process: Will process {len(prompts)} prompts")

        # Create a sampling params object.
        sampling_params = SamplingParams(
            temperature=0.0, prompt_logprobs=0, max_tokens=1
        )

        # Create an LLM.
        logger.info(f"Initializing LLM...")
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            **vllm_config,
        )
        
        logger.info(f"Single process: Generating outputs...")
        outputs = llm.generate(prompts, sampling_params)
        
        logger.info(f"Single process: Processing {len(outputs)} outputs...")
        results = []
        for i, output in enumerate(outputs):
            logprobs = process_vllm_logprobs(output.prompt_logprobs)
            avg_logprob = np.mean(logprobs)
            perplexity = np.exp(-avg_logprob)
            results.append({"original_index": dataset.index[i], "logprobs": logprobs, \
                "avg_logprob": avg_logprob, "perplexity": perplexity, "num_tokens": len(logprobs)})

        logger.info(f"Single process: Completed processing")
    
        return results
        
    except Exception as e:
        logger.error(f"Single process: ERROR - {str(e)}")
        raise e


def _tokenize_batch(texts_batch, tokenizer):
    """Helper function to tokenize a batch of texts efficiently."""
    try:
        # Tokenize all texts in the batch at once for efficiency
        encoded = tokenizer(texts_batch, add_special_tokens=False, return_attention_mask=False)
        return [len(tokens) for tokens in encoded['input_ids']]
    except Exception:
        # Fallback to individual tokenization if batch fails
        return [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts_batch]


def load_and_filter_dataset(dataset_path: str, model: str, prompt_column: str, max_len: int = 10_000) -> pd.DataFrame:
    """
    Load a dataset and filter it based on token count using the model's tokenizer.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset file (csv or parquet)
    model : str
        Model name/path to load the appropriate tokenizer
    prompt_column : str
        Name of the column containing the text to filter
    max_len : int
        Maximum number of tokens allowed per text (default: 10,000)
        
    Returns
    -------
    pd.DataFrame
        Filtered dataset with texts having <= max_len tokens
    """
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    logger.info(f"Dataset loaded with {len(df)} rows")

    if max_len == -1:
        logger.info(f"No filtering: Returning the entire dataset")
        return df
    
    # Load tokenizer for the model
    logger.info(f"Loading tokenizer for model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    # Get all texts to process
    texts = df[prompt_column].tolist()
    logger.info(f"Tokenizing {len(texts)} texts for filtering...")
    
    # Process in batches for efficiency
    batch_size = 1000  # Process 1000 texts at a time
    token_counts = []
    
    # Use ThreadPoolExecutor for parallel processing of batches
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        # Process batches in parallel
        tokenize_func = partial(_tokenize_batch, tokenizer=tokenizer)
        batch_results = list(executor.map(tokenize_func, batches))
        
        # Flatten results
        for batch_result in batch_results:
            token_counts.extend(batch_result)
    
    # Add token counts to dataframe and filter
    df['token_count'] = token_counts
    original_count = len(df)
    df = df[df['token_count'] <= max_len]
    filtered_count = len(df)
    
    logger.info(f"Filtered dataset: {original_count} -> {filtered_count} rows "
                f"(removed {original_count - filtered_count} rows with > {max_len} tokens)")
    
    # Drop the temporary token_count column
    df = df.drop('token_count', axis=1)
    
    return df


def save_logprob_results(results: list, save_path: str):
    """Save the logprob results to a parquet file."""
    results_df = pd.DataFrame(results)
    results_df.to_parquet(save_path)


# main function that handles the entire process
def evaluate_document(
    model: str,
    tp_size: int,
    vllm_config: dict,
    dataset_path: str,
    save_path: str,
    prompt_column: str = "text",
    max_len: int = 10_000,
    verbose: bool = False,
):
    """
    Evaluate documents using vLLM with token-based filtering.
    
    Parameters
    ----------
    model : str
        Model name/path for evaluation and tokenization
    tp_size : int
        Tensor parallelism size
    vllm_config : dict
        vLLM configuration parameters
    dataset_path : str
        Path to the dataset file
    save_path : str
        Path to save evaluation results
    prompt_column : str
        Name of the text column to evaluate (default: "text")
    max_len : int
        Maximum number of tokens per text (default: 10,000)
    verbose : bool
        Enable verbose logging (default: False)
    """
    if verbose:
        set_all_logging_level(logging.INFO, to_stdout=True)
    else:
        set_all_logging_level(logging.WARNING, to_stdout=True)

    # load and filter the dataset using token-based filtering
    df = load_and_filter_dataset(dataset_path, model, prompt_column, max_len)

    # process the dataset
    results = vllm_process_documents(model, tp_size, vllm_config, df, prompt_column)
    
    # save the results
    save_logprob_results(results, save_path)


if __name__ == "__main__":
    import argparse
    import os, torch
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp_size", type=int, required=True)
    parser.add_argument("--vllm_config_path", type=str, default="vllm_config.yaml")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--prompt_column", type=str, default="text")
    parser.add_argument("--max_len", type=int, default=-1, 
                        help="Maximum number of tokens per text (default: -1, no filtering)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    vllm_config = load_vllm_config(args.vllm_config_path)

    # construct save path from args
    model_name = args.model.split("/")[-1]
    yyyy_mm_dd = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(args.save_folder, f"{model_name}_{yyyy_mm_dd}.parquet")

    # verify the `tp_size` is valid
    if torch.cuda.is_available():
        logger.info(f"Using CUDA devices: {torch.cuda.device_count()}")
        tp_size = min(torch.cuda.device_count(), args.tp_size)
    else:
        logger.warning("No CUDA devices found. Using CPU.")
        tp_size = 1

    logger.info(f"Effective TP size: {tp_size} (out of {args.tp_size} requested)")

    evaluate_document(
        model=args.model,
        tp_size=tp_size,
        vllm_config=vllm_config,
        dataset_path=args.dataset_path,
        save_path=save_path,
        prompt_column=args.prompt_column,
        max_len=args.max_len,
        verbose=args.verbose
    )