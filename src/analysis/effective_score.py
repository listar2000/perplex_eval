"""
Utilize the bits-per-byte (bpb) information to take different LLMs log probs on the same text,
normalize into bpb for the sequence of bytes, and obtain per-byte sample variances. These sample 
variances are then used to filter out parts of the text that differentiate the LLMs. We finally 
would use these parts to compute the `effective score` for each LLM on this text.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import sys
from pydantic import BaseModel
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer

from src.evaluate.vllm_utils import LLM_TO_VLLM_ID_MAPPING


class EffectiveScores(BaseModel):
    # Per model
    effective_avg_logprob_per_byte: Dict[str, float]      # average log p (per byte) over selected region
    effective_bpb: Dict[str, float]                       # bits per byte = -avg_logprob_per_byte / ln 2
    effective_ppl_per_byte: Dict[str, float]              # exp(-avg_logprob_per_byte)


def _char_to_byte_index_map(text: str) -> np.ndarray:
    """Map each character boundary to its UTF-8 byte index.
    Returns an array `cb` of length len(text)+1 such that the byte span of text[i:j] is [cb[i], cb[j])."""
    b = text.encode("utf-8")
    cb = np.zeros(len(text) + 1, dtype=np.int32)
    # Walk characters, accumulate bytes
    pos_bytes = 0
    for i, ch in enumerate(text):
        cb[i] = pos_bytes
        pos_bytes += len(ch.encode("utf-8"))
    cb[len(text)] = pos_bytes
    return cb


def _per_byte_density_from_tokens(
    text: str,
    token_logprobs: List[float],
    tokenizer,  # HF fast tokenizer
    char2byte: np.ndarray = None
) -> np.ndarray:
    """
    Build per-byte NLL density for the entire original text.
    Assumes token_logprobs is one shorter than the #tokens (no logprob for the 1st token).
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = enc["offset_mapping"]  # list[(start_char, end_char)]
    # Expect: len(token_logprobs) == len(offsets) - 1
    skip_num = len(offsets) - len(token_logprobs)
    if skip_num > 1:
        raise ValueError(
            f"Mismatch: got {len(token_logprobs)} log_probs but tokenizer produced {len(offsets)} tokens."
        )

    # 1) Global char->byte map on the ORIGINAL text (no trimming)
    n_bytes = int(char2byte[-1])
    nll_density = np.zeros(n_bytes, dtype=np.float64)

    # 2) Pair each logprob with the *second and later* tokens' spans
    #    i.e., zip(logprobs, offsets[1:])
    for lp, (cs, ce) in zip(token_logprobs, offsets[skip_num:]):
        # Guard weird zero/invalid spans
        if not (0 <= cs <= ce <= len(text)):
            continue
        if ce <= cs:
            continue

        bs = int(char2byte[cs])  # byte start in original text
        be = int(char2byte[ce])  # byte end in original text
        length_bytes = max(1, be - bs)

        nll = -float(lp)  # negative log-prob
        # Uniformly spread the token's NLL over its covered bytes
        nll_density[bs:be] += nll / length_bytes

    return nll_density



def _bin_by_bytes(density: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-byte density into K relative-position bins.
    Returns (bin_means, bin_edges_rel)."""
    n_bytes = len(density)
    if n_bytes == 0:
        return np.zeros(K), np.linspace(0, 1, K + 1)
    # Compute edges in byte coordinates
    edges = np.linspace(0, n_bytes, K + 1, dtype=np.float64)
    bin_means = np.zeros(K, dtype=np.float64)
    for i in range(K):
        a = int(edges[i])
        b = int(edges[i + 1]) if i + 1 < len(edges) else n_bytes
        if b <= a:
            bin_means[i] = 0.0
        else:
            bin_means[i] = density[a:b].mean()
    return bin_means, (edges / n_bytes)


def find_disagreement_and_effective_scores(
    log_probs: Dict[str, List[float]],
    tokenizers: Dict[str, AnyTokenizer],
    original_text: str,
    *,
    K: int = 512,
    # How to focus aggregation:
    #   - mode="topq": average only over top-q variance bins
    #   - mode="weighted": average over all bins with weights ~ variance^alpha
    #   - mode="uniform": average over all bins equally (i.e., standard bpb)
    mode: str = "topq",
    q: float = 0.2,            # top 20% highest-variance bins if mode="topq"
    alpha: float = 1.0,        # weight exponent if mode="weighted"
    center_per_model: bool = False,   # z-score across bins per model before variance, to reduce calibration shift
) -> EffectiveScores:
    """
    Full pipeline:
      1) Build per-byte NLL density per model from token log-probs (tokenizer-invariant).
      2) Resample to K relative-position bins.
      3) (Optional) Per-model centering to emphasize relative spikes.
      4) Compute across-model variance per bin; smooth.
      5) Select bins (top-q or weighted); aggregate each model's log-probs accordingly.
    Returns effective scores and disagreement diagnostics.
    """
    # 0) Precompute char->byte map
    c2b = _char_to_byte_index_map(original_text)
    n_bytes = c2b[-1]
    if n_bytes == 0:
        raise ValueError("original_text is empty after UTF-8 encoding.")

    # 1) Per-byte NLL density for each model
    per_model_density = {}
    for name, lps in log_probs.items():
        if name not in tokenizers:
            raise KeyError(f"Missing tokenizer for model '{name}'.")
        nll_density = _per_byte_density_from_tokens(original_text, lps, tokenizers[name], c2b)
        per_model_density[name] = nll_density

    # 2) Bin to common grid
    bin_means = {}            # per model: shape (K,)
    for name, dens in per_model_density.items():
        bmean, edges_rel = _bin_by_bytes(dens, K)
        bin_means[name] = bmean

    bin_edges_rel = edges_rel  # same for all

    # 3) Optionally center/z-score per model (across bins) to reduce level shifts
    X = np.stack([bin_means[name] for name in bin_means.keys()], axis=0)  # [M, K]
    if center_per_model:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-8
        X_for_var = (X - mu) / sd
    else:
        X_for_var = X

    # 4) Variance across models per bin; smooth
    var_bins_raw = X_for_var.var(axis=0, ddof=1) if X_for_var.shape[0] > 1 else np.zeros(X_for_var.shape[1]) # shape: (K,)
    var_bins = var_bins_raw  # TODO: consider gaussian smoothing, if needed

    # 5) Select bins and aggregate
    if mode == "topq":
        q = float(np.clip(q, 1e-6, 0.999999))
        thresh = np.quantile(var_bins, 1.0 - q)
        selected_mask = var_bins >= thresh
        # Avoid degenerate all-False (can happen with tiny q)
        if not selected_mask.any():
            selected_mask[np.argmax(var_bins)] = True
        weights = selected_mask.astype(np.float64)
    elif mode == "weighted":
        weights = np.power(np.maximum(var_bins, 0.0), alpha)
        if weights.max() <= 0:
            # fallback to uniform
            weights = np.ones_like(weights)
        selected_mask = weights > 0
    elif mode == "uniform":
        weights = np.ones_like(var_bins)
        selected_mask = weights > 0
    else:
        raise ValueError(f"Unknown mode='{mode}'. Choose 'topq', 'weighted', or 'uniform'.")

    # Normalize weights over *selected* bins
    weights = weights * (selected_mask.astype(np.float64)) # shape: (K,)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    # Effective averages per model (per-byte logprob density)
    effective_avg_logprob_per_byte = {}
    effective_bpb = {}
    effective_ppl_per_byte = {}

    # Note: X holds per-bin MEAN of per-byte NLL (positive).
    # We want average logprob per byte, so use logprob = -NLL.
    # Average over bins using `weights`, then convert to bpb & ppl.
    # Per model’s per-bin mean NLL:
    X_nll = np.stack([bin_means[name] for name in bin_means.keys()], axis=0)  # shape: (M, K)
    # Weighted mean NLL (per byte) over selected bins:
    nll_bar = (X_nll * weights[None, :]).sum(axis=1)  # shape: (M,)

    for i, name in enumerate(bin_means.keys()):
        avg_logp_per_byte = -float(nll_bar[i])
        effective_avg_logprob_per_byte[name] = avg_logp_per_byte
        bpb = float(nll_bar[i] / math.log(2.0))      # bits per byte
        effective_bpb[name] = bpb
        effective_ppl_per_byte[name] = float(math.exp(nll_bar[i]))  # exp(NLL_per_byte)

    return EffectiveScores(
        effective_avg_logprob_per_byte=effective_avg_logprob_per_byte,
        effective_bpb=effective_bpb,
        effective_ppl_per_byte=effective_ppl_per_byte
    )


def fetch_all_tokenizers(llm_names: List[str]) -> Dict[str, AnyTokenizer]:
    """
    Fetch all tokenizers using the `vllm` built-in `get_tokenizer` function.
    """
    tokenizers = {}
    for llm_name in llm_names:
        tokenizer = get_tokenizer(LLM_TO_VLLM_ID_MAPPING[llm_name])
        tokenizers[llm_name] = tokenizer
    return tokenizers


# local testing
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    merged_df = pd.read_parquet("output/merged/merged_results_logprobs.parquet")
    # for retrieving original text
    text_df = pd.read_parquet("data/redpajama-subset-chunked.parquet")

    # LLM names are column names that are also in `LLM_TO_VLLM_ID_MAPPING`
    col_names = merged_df.columns
    print("Column names in the merged dataframe: ", col_names)
    llm_names = [name for name in col_names if name in LLM_TO_VLLM_ID_MAPPING]
    print("LLM names in the merged dataframe: ", llm_names)

    # get the tokenizers ready
    tokenizers = fetch_all_tokenizers(llm_names)

    # now iterate over the rows to obtain the effective_scores, and store them
    effective_avg_logprobs, effective_bpb, effective_ppl = [], [], []

    for index, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        original_index = row["original_index"]
        text = text_df.iloc[original_index]["text"]
        log_probs = {name: row[name] for name in llm_names}

        effective_scores = find_disagreement_and_effective_scores(log_probs, tokenizers, text, K=512, mode="topq", q=0.1)
        effective_avg_logprobs.append(effective_scores.effective_avg_logprob_per_byte)
        effective_bpb.append(effective_scores.effective_bpb)
        effective_ppl.append(effective_scores.effective_ppl_per_byte)
        
    # finally save the results into three different parquet files, each should have the same columns as the merged_df
    # Add "original_index" column to each DataFrame
    effective_avg_logprobs_df = pd.DataFrame(effective_avg_logprobs, columns=llm_names)
    effective_avg_logprobs_df["original_index"] = merged_df["original_index"].values

    effective_bpb_df = pd.DataFrame(effective_bpb, columns=llm_names)
    effective_bpb_df["original_index"] = merged_df["original_index"].values

    effective_ppl_df = pd.DataFrame(effective_ppl, columns=llm_names)
    effective_ppl_df["original_index"] = merged_df["original_index"].values

    effective_avg_logprobs_df.to_parquet("output/filtered/effective_avg_logprobs.parquet")
    effective_bpb_df.to_parquet("output/filtered/effective_bpb.parquet")
    effective_ppl_df.to_parquet("output/filtered/effective_ppl.parquet")
    
    