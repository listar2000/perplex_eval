"""
This file contains constants for the data module.
"""
from functools import partial
import ast
import tldextract
from pydantic import BaseModel
from typing import Callable

LOCAL_DATA_DIR = "./data"

def generic_extract_domain_from_url(sample: dict, url_column: str) -> str | None:
    """
    Extracts the registered domain (domain + suffix) from any URL.
    E.g. 'sub.blog.example.co.uk' → 'example.co.uk'.
    """
    if url_column not in sample or not sample[url_column]:
        return None
    try:
        url = sample[url_column]
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    except Exception:
        return None

def redpajama_extract_domain_fn(sample: dict) -> str | None:
    """
    Specialized function for RedPajama dataset.
    """
    meta = sample["meta"]
    
    # Handle case where meta is already a dict/struct (streaming datasets)
    if isinstance(meta, dict):
        meta_dict = meta
    else:
        # Handle case where meta is a JSON string (non-streaming datasets)
        try:
            meta_dict = ast.literal_eval(meta)
        except ValueError:
            print(f"Error parsing meta: {meta}")
            return "Unknown"
    
    # rule No.1: look for `source` in the meta dict
    if "source" in meta_dict:
        raw_source = meta_dict["source"]
        if raw_source.startswith("cc"):
            return "commoncrawl"
        else:
            return raw_source
    # rule No.2: if `arxiv_id` is in meta dict, this is from arxiv
    if "arxiv_id" in meta_dict:
        return "arxiv"
    
    return "Unknown"
    
class DatasetConfig(BaseModel):
    repo_id: str
    split: str
    extract_domain_fn: Callable[[dict], str | None]

DATASET_CONFIGS = {
    "refinedweb": DatasetConfig(
        repo_id="tiiuae/falcon-refinedweb",
        split="train",
        extract_domain_fn=partial(generic_extract_domain_from_url, url_column="url"),
    ),
    "c4": DatasetConfig(
        repo_id="allenai/c4",
        split="en",
        extract_domain_fn=partial(generic_extract_domain_from_url, url_column="url"),
    ),
    "redpajama-1t-sample": DatasetConfig(
        repo_id="togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
        extract_domain_fn=redpajama_extract_domain_fn,
    ),
}