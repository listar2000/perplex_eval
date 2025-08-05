"""
This file helps stream large pretraining datasets from Hugging Face, and apply subsetting to obtain
a much smaller dataset.
"""
from collections import Counter, defaultdict
from functools import partial
import os
import multiprocessing

from datasets import Dataset, IterableDataset, load_dataset, Features, Value
import tldextract
import tqdm
import json

from utils import DATASET_CONFIGS, DatasetConfig, LOCAL_DATA_DIR

def subset_pretrain_data(
    dataset_config: DatasetConfig,
    num_domains: int = 100,
    docs_per_domain: int = 1_000,
    get_iterable: bool = False,
    add_domain_column: bool = False,
    save_folder: str = LOCAL_DATA_DIR,
    num_workers: int = 8,
    seed: int = 42,
) -> Dataset | IterableDataset:
    """
    Subset a pretraining dataset from Hugging Face.

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

    Returns
    -------
    Dataset or IterableDataset
        A Dataset or IterableDataset containing the subsetted dataset.
    """
    num_workers = min(num_workers, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} workers")

    # Try to load without strict schema validation to handle varying meta structures
    dataset = load_dataset(
        dataset_config.repo_id,
        split=dataset_config.split,
        streaming=False,
        trust_remote_code=True
    )

    # step 1: Count domains over the entire dataset
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

    # reload again to get the full dataset
    dataset = load_dataset(
        dataset_config.repo_id,
        split=dataset_config.split,
        streaming=False,
        trust_remote_code=True,
    ).shuffle(seed=seed)

    collected = Counter()
    filled_domains = set()
    subset_samples = []

    # This loop is sequential but will exit early, so it's much faster.
    for sample in tqdm.tqdm(dataset, desc="Collecting samples"):
        domain = dataset_config.extract_domain_fn(sample)
        if domain in top_domains and domain not in filled_domains:
            if add_domain_column:
                sample["domain"] = domain

            subset_samples.append(sample)
            collected[domain] += 1
            
            if collected[domain] >= docs_per_domain:
                filled_domains.add(domain)
                print(f"Completed collecting for domain: {domain} ({len(filled_domains)}/{num_domains})")

            if len(filled_domains) == num_domains:
                print("Finished collecting all required samples.")
                break

    # step 4: Save the dataset
    subsample_size = len(subset_samples)
    save_path = os.path.join(save_folder, f"{dataset_config.repo_id.split('/')[-1]}_{subsample_size}.csv")

    subset_dataset: Dataset = Dataset.from_list(subset_samples)
    subset_dataset.to_csv(save_path)

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
    # use the saved csv file to load the dataset
    redpajama_dataset = Dataset.from_csv(
        "/net/scratch2/listar2000/perplex_eval/data/RedPajama-Data-1T-Sample_51510.csv",
    )
    push_to_hub(redpajama_dataset, "listar2000", "redpajama-subset-50k", description="A 50k subset of the RedPajama-1T dataset")