"""
This file helps stream large pretraining datasets from Hugging Face, and apply subsetting to obtain
a much smaller dataset.
"""
from collections import Counter, defaultdict
from functools import partial
import os
import multiprocessing
import random
import re
from typing import List, Tuple

from datasets import Dataset, IterableDataset, load_dataset, Features, Value
import tldextract
import tqdm
import json
from transformers import AutoTokenizer

from utils import DATASET_CONFIGS, DatasetConfig, LOCAL_DATA_DIR


# Global tokenizer for worker processes (to avoid reloading)
_worker_tokenizer = None
_worker_tokenizer_name = None

def _init_worker_tokenizer(tokenizer_name):
    """Initialize tokenizer in worker process (called once per worker)."""
    global _worker_tokenizer, _worker_tokenizer_name
    if _worker_tokenizer_name != tokenizer_name:
        from transformers import AutoTokenizer
        _worker_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if _worker_tokenizer.pad_token is None:
            _worker_tokenizer.pad_token = _worker_tokenizer.eos_token
        _worker_tokenizer_name = tokenizer_name

def _chunking_worker(args):
    """
    Worker function for multiprocessing chunking.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (sample, original_id, domain, tokenizer_name, max_tokens, text_col)
        
    Returns
    -------
    List[dict]
        List of processed chunks for the sample.
    """
    sample, original_id, domain, tokenizer_name, max_tokens, text_col = args
    overlap_tokens = 0
    
    # Use the global tokenizer (initialized once per worker)
    _init_worker_tokenizer(tokenizer_name)
    
    return process_sample_with_chunking(
        sample, original_id, domain, _worker_tokenizer, max_tokens, overlap_tokens, text_col
    )


def chunk_text_by_tokens(
    text: str, 
    tokenizer, 
    max_tokens: int = 4096, 
    overlap_tokens: int = 0
) -> List[str]:
    """
    Split text into chunks with a maximum number of tokens per chunk.
    
    Parameters
    ----------
    text : str
        The input text to chunk.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for counting tokens.
    max_tokens : int, optional
        Maximum number of tokens per chunk (default is 4096).
    overlap_tokens : int, optional
        Number of tokens to overlap between chunks (default is 0).
        
    Returns
    -------
    List[str]
        List of text chunks.
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Decode the chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Try to break at sentence boundaries to avoid breaking mid-sentence
        if end_idx < len(tokens):  # Not the last chunk
            # Look for sentence endings in the last portion of the chunk
            sentences = re.split(r'[.!?]\s+', chunk_text)
            if len(sentences) > 1:
                # Keep all complete sentences except the last incomplete one
                complete_sentences = sentences[:-1]
                chunk_text = '. '.join(complete_sentences)
                if chunk_text and not chunk_text.endswith(('.', '!', '?')):
                    chunk_text += '.'
                
                # Re-tokenize to get actual token count for this adjusted chunk
                actual_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                
                # Ensure we always make progress, even if sentence boundary adjustment fails
                if len(actual_tokens) > 0:
                    start_idx += len(actual_tokens) - overlap_tokens
                else:
                    # Fallback: advance by at least 1 token to prevent infinite loop
                    start_idx += max(1, max_tokens - overlap_tokens)
            else:
                start_idx = end_idx - overlap_tokens
        else:
            start_idx = end_idx
            
        if chunk_text.strip():  # Only add non-empty chunks
            chunks.append(chunk_text.strip())
            
    return chunks


def process_sample_with_chunking(
    sample: dict,
    original_id: int,
    domain: str,
    tokenizer,
    max_tokens: int = 4096,
    overlap_tokens: int = 0,
    text_col: str = "text"
) -> List[dict]:
    """
    Process a single sample and create chunks if necessary.
    
    Parameters
    ----------
    sample : dict
        The original sample from the dataset.
    original_id : int
        The original index of the sample before shuffling.
    domain : str
        The domain of the sample.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for chunking.
    max_tokens : int, optional
        Maximum number of tokens per chunk (default is 4096).
    text_col : str, optional
        Name of the text column (default is "text").
        
    Returns
    -------
    List[dict]
        List of processed samples (chunks).
    """
    text = sample[text_col]
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens, overlap_tokens)
    
    processed_samples = []
    for chunk_id, chunk_text in enumerate(chunks):
        processed_sample = {
            "original_id": original_id,
            "chunk_id": chunk_id,
            "domain": domain,
            "text": chunk_text
        }
        processed_samples.append(processed_sample)
    
    return processed_samples


def subset_pretrain_data(
    dataset_config: DatasetConfig,
    num_domains: int = 100,
    docs_per_domain: int = 1_000,
    get_iterable: bool = False,
    save_folder: str = LOCAL_DATA_DIR,
    num_workers: int = 8,
    seed: int = 42,
    tokenizer_name: str = "01-ai/Yi-1.5-6B-Chat",
    max_tokens: int = 4096,
    text_col: str = "text",
    enable_chunking: bool = True,
    chunk_batch_size: int = 100,
) -> Dataset | IterableDataset:
    """
    Subset a pretraining dataset from Hugging Face with optional text chunking.

    Parameters
    ----------
    dataset_config : DatasetConfig
        The configuration for the dataset to subset.
    num_domains : int, optional
        The number of domains to collect (default is 100).
    docs_per_domain : int, optional
        The number of documents to collect per domain (default is 1,000).
    get_iterable : bool, optional
        Whether to return an IterableDataset (default is False).
    add_domain_column : bool, optional
        Whether to add a column for the domain to the dataset (default is False).
    save_folder : str, optional
        The folder to save the subsetted dataset (default is LOCAL_DATA_DIR).
    num_workers : int, optional
        The number of workers to use for parallel processing (default is 8).
    seed : int, optional
        The random seed to use for shuffling the dataset (default is 42).
    tokenizer_name : str, optional
        The name of the tokenizer to use for chunking (default is "01-ai/Yi-1.5-6B-Chat").
    max_tokens : int, optional
        Maximum number of tokens per chunk (default is 4096).
    text_col : str, optional
        Name of the text column in the dataset (default is "text").
    enable_chunking : bool, optional
        Whether to enable text chunking (default is True).
    chunk_batch_size : int, optional
        Number of samples to process in each batch for parallel chunking (default is 100).

    Returns
    -------
    Dataset or IterableDataset
        A Dataset or IterableDataset containing the subsetted and chunked dataset.
        Each row contains: original_id, chunk_id, domain, text.
    """
    num_workers = min(num_workers, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers")
    
    # Initialize tokenizer if chunking is enabled
    tokenizer = None
    if enable_chunking:
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load dataset without shuffling first (to track original indices)
    print("Loading dataset...")
    dataset = load_dataset(
        dataset_config.repo_id,
        split=dataset_config.split,
        streaming=False,
        trust_remote_code=True
    )

    # step 1: Count domains over the entire dataset
    print("Counting domains...")
    domain_counts = Counter()
    
    with multiprocessing.Pool(num_workers) as pool:
        results_iterator = pool.imap_unordered(
            dataset_config.extract_domain_fn,
            dataset,
            chunksize=1000
        )
        for domain in tqdm.tqdm(results_iterator, desc="Counting domains"):
            if domain is not None:
                domain_counts[domain] += 1

    # step 2: Pick the top num_domains domains
    top_domain_stats = domain_counts.most_common(num_domains)
    print("Top domain stats: ", top_domain_stats)
    top_domains = {d for d, _ in top_domain_stats}

    # step 3: Create a mapping of original indices before shuffling
    print("Creating index mapping...")
    original_indices = list(range(len(dataset)))
    
    # Create shuffled indices
    random.seed(seed)
    shuffled_indices = original_indices.copy()
    random.shuffle(shuffled_indices)

    # step 4: Collect samples using shuffled order but track original indices
    collected = Counter()
    filled_domains = set()
    all_processed_samples = []
    samples_to_process = []  # Buffer for samples to process in batches

    print("Collecting samples...")
    for shuffled_idx in tqdm.tqdm(shuffled_indices, desc="Collecting samples"):
        sample = dataset[shuffled_idx]
        original_id = shuffled_idx  # This is the original index before shuffling
        
        domain = dataset_config.extract_domain_fn(sample)
        if domain in top_domains and domain not in filled_domains:
            
            # Add to batch for processing
            samples_to_process.append((sample, original_id, domain))
            collected[domain] += 1
            
            if collected[domain] >= docs_per_domain:
                filled_domains.add(domain)
                print(f"Completed collecting for domain: {domain} ({len(filled_domains)}/{num_domains})")

            if len(filled_domains) == min(num_domains, len(top_domains)):
                print(f"Finished collecting {len(samples_to_process)} samples for {len(filled_domains)} domains.")
                break

    print(f"Collected {len(samples_to_process)} samples. Processing with chunking...")
    
    # step 5: Process samples with multiprocessing if chunking is enabled
    if enable_chunking and samples_to_process:
        # Prepare arguments for multiprocessing
        processing_args = [
            (sample, original_id, domain, tokenizer_name, max_tokens, text_col)
            for sample, original_id, domain in samples_to_process
            if text_col in sample
        ]
        
        # Process in batches with multiprocessing
        with multiprocessing.Pool(num_workers) as pool:
            # Use imap_unordered for better memory efficiency with large datasets
            results_iterator = pool.imap_unordered(
                _chunking_worker, 
                processing_args, 
                chunksize=chunk_batch_size
            )
            
            # Process results as they become available
            for result in tqdm.tqdm(results_iterator, total=len(processing_args), desc="Processing chunks"):
                all_processed_samples.extend(result)
        
        # Handle samples without the text column (fallback)
        for sample, original_id, domain in samples_to_process:
            if text_col not in sample:
                processed_sample = {
                    "original_id": original_id,
                    "chunk_id": 0,
                    "domain": domain,
                    "text": ""
                }
                all_processed_samples.append(processed_sample)
                
    else:
        # Process without chunking
        print("Processing without chunking...")
        for sample, original_id, domain in tqdm.tqdm(samples_to_process, desc="Processing samples"):
            processed_sample = {
                "original_id": original_id,
                "chunk_id": 0,
                "domain": domain,
                "text": sample.get(text_col, "")
            }
            all_processed_samples.append(processed_sample)

    # step 6: Save the dataset
    total_chunks = len(all_processed_samples)
    save_path = os.path.join(save_folder, f"{dataset_config.repo_id.split('/')[-1]}_{total_chunks}_chunks.parquet")

    print(f"Creating dataset with {total_chunks} chunks from {sum(collected.values())} original documents")
    subset_dataset: Dataset = Dataset.from_list(all_processed_samples)
    subset_dataset.to_parquet(save_path)
    print(f"Dataset saved to: {save_path}")

    if get_iterable:
        return subset_dataset.to_iterable_dataset()
    else:
        return subset_dataset


def push_to_hub(dataset: Dataset, username: str, repo_name: str, description: str = None, message: str = None):
    """
    Push a dataset to the Hugging Face Hub.
    Make sure `huggingface-cli login` is run before this function.
    """
    dataset.push_to_hub(f"{username}/{repo_name}", commit_description=description, commit_message=message)


if __name__ == "__main__":
    # Example usage with chunking enabled
    from utils import DATASET_CONFIGS
    
    # Subset RedPajama with chunking
    # dataset_config = DATASET_CONFIGS["redpajama-1t-sample"]
    
    # subset_dataset = subset_pretrain_data(
    #     dataset_config=dataset_config,
    #     num_domains=10,  # Smaller number for testing
    #     docs_per_domain=6000,  # Smaller number for testing
    #     enable_chunking=True,
    #     max_tokens=2048,
    #     text_col="text",
    #     tokenizer_name="01-ai/Yi-1.5-6B-Chat",
    #     chunk_batch_size=1000,
    #     num_workers=16
    # )
    from datasets import load_dataset
    subset_dataset = load_dataset("parquet", data_files="data/redpajama-subset-chunked.parquet")
    
    print(f"Created dataset with {len(subset_dataset)} chunks")
    print("Sample from dataset:")
    print(subset_dataset["train"][0])
    
    # Optionally push to hub
    push_to_hub(subset_dataset, "listar2000", "redpajama-subset-chunked-2048", 
                description="A chunked subset of the RedPajama-1B samples with maximum 2048 tokens per chunk.")