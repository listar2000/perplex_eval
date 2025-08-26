"""
Merge LLM inference results from multiple parquet files.

This script takes multiple parquet files containing LLM inference results and merges
them into a single file by finding the common subset of documents (by original_index)
and creating a merged dataframe with one column per LLM for the specified metric.

Usage:
```bash
python -m src.evaluate.merge_llm_results \
    --input_folder output/raw \
    --output_path output/merged/merged_results \
    --merge_column logprobs \
    --verbose
```
"""

import logging
import os
import pandas as pd
from glob import glob
from typing import List, Dict, Set, Optional
import numpy as np
from vllm_utils import LLM_TO_ALIAS_MAPPING

logger = logging.getLogger(__name__)


def extract_model_name_from_filename(filename: str) -> str:
    """Extract the model name from a parquet filename."""
    basename = os.path.basename(filename)
    # Remove the date suffix and .parquet extension
    # Format: ModelName_YYYY-MM-DD.parquet
    model_name = basename.rsplit('_', 1)[0]
    return model_name


def find_common_indices(parquet_files: List[str]) -> Set[int]:
    """Find the intersection of original_index values across all parquet files."""
    logger.info(f"Finding common indices across {len(parquet_files)} files...")
    
    all_indices = []
    for filepath in parquet_files:
        df = pd.read_parquet(filepath)
        indices = set(df['original_index'])
        all_indices.append(indices)
        logger.debug(f"{os.path.basename(filepath)}: {len(indices)} indices")
    
    common_indices = set.intersection(*all_indices)
    logger.info(f"Found {len(common_indices)} common indices")
    return common_indices


def merge_llm_results(
    input_folder: str,
    merge_column: str,
    llm_name_mapping: Optional[Dict[str, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Merge LLM inference results from multiple parquet files.
    
    Args:
        input_folder: Path to folder containing parquet files
        merge_column: Column to merge on ('perplexity', 'avg_logprob', 'num_tokens')
        llm_name_mapping: Optional mapping from model names to column names
        verbose: Whether to enable verbose logging
        
    Returns:
        Merged DataFrame with original_index and one column per LLM
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    if llm_name_mapping is None:
        llm_name_mapping = LLM_TO_ALIAS_MAPPING
    
    # Find all parquet files
    pattern = os.path.join(input_folder, "*.parquet")
    parquet_files = glob(pattern)
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_folder}")
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Validate merge column
    valid_columns = ['perplexity', 'avg_logprob', 'num_tokens', "logprobs"]
    if merge_column not in valid_columns:
        raise ValueError(f"merge_column must be one of {valid_columns}, got {merge_column}")
    
    # Find common indices
    common_indices = find_common_indices(parquet_files)
    
    # Create the merged dataframe
    merged_data = {'original_index': sorted(list(common_indices))}
    
    logger.info(f"Merging {len(parquet_files)} files on column '{merge_column}'...")
    
    for filepath in parquet_files:
        # Extract model name
        model_name = extract_model_name_from_filename(filepath)
        
        # Get clean column name
        clean_name = llm_name_mapping.get(model_name, model_name)
        
        # Load data and filter to common indices
        df = pd.read_parquet(filepath)
        df_filtered = df[df['original_index'].isin(common_indices)].copy()
        
        # Sort by original_index to ensure consistent ordering
        df_filtered = df_filtered.sort_values('original_index')
        
        # special case, we convert to a list of float32
        if merge_column == "logprobs":
            df_filtered[merge_column] = df_filtered[merge_column].apply(lambda x: np.array(x, dtype=np.float32))
        
        # Add the column data
        merged_data[clean_name] = df_filtered[merge_column].values
        
        logger.info(f"Added {clean_name}: {len(df_filtered)} rows")
    
    # Create merged dataframe
    merged_df = pd.DataFrame(merged_data)
    
    logger.info(f"Created merged dataframe with shape {merged_df.shape}")
    logger.info(f"Columns: {list(merged_df.columns)}")
    
    return merged_df


def save_merged_results(
    merged_df: pd.DataFrame, 
    output_path: str, 
    merge_column: str,
    to_csv: bool = False
) -> None:
    """Save the merged results as both parquet and CSV files."""
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add merge column info to filename
    base_path = f"{output_path}_{merge_column}"
    
    
    # Save as CSV
    if to_csv:
        csv_path = f"{base_path}.csv"
        merged_df.to_csv(csv_path, index=False)
        logger.info(f"Saved merged CSV file: {csv_path}")
    else: # Save as parquet
        parquet_path = f"{base_path}.parquet"
        merged_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved merged parquet file: {parquet_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge LLM inference results")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing parquet files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path (without extension)")
    parser.add_argument("--merge_column", type=str, default="perplexity",
                        choices=["perplexity", "avg_logprob", "num_tokens", "logprobs"],
                        help="Column to merge on")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Merge the results
    merged_df = merge_llm_results(
        input_folder=args.input_folder,
        merge_column=args.merge_column,
        verbose=args.verbose
    )
    
    # Save the results
    save_merged_results(merged_df, args.output_path, args.merge_column)
    
    print(f"Merging complete!")
    print(f"Merged {merged_df.shape[0]} documents across {merged_df.shape[1]-1} LLMs")
    print(f"Files saved with base name: {args.output_path}_{args.merge_column}") 