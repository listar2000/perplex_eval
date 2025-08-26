"""
Utility functions for vLLM inference.
"""
import yaml
import os
from vllm.sequence import Logprob


LLM_TO_ALIAS_MAPPING = {
    "deepseek-llm-7b-chat": "DeepSeek-7B-Chat",
    "Yi-1.5-6B-Chat": "Yi-1.5-6B-Chat", 
    "Yi-1.5-9B-Chat-16K": "Yi-1.5-9B-Chat-16K",
    "Qwen3-0.6B": "Qwen3-0.6B",
    "mpt-7b-instruct": "MPT-7B-Instruct",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct", 
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Phi-4": "Phi-4",
    "Falcon3-7B-Instruct": "Falcon3-7B-Instruct",
    "OLMo-2-0425-1B-Instruct": "OLMo-2-1B-Instruct",
    "OLMo-2-1124-7B-Instruct": "OLMo-2-7B-Instruct",
    "pythia-1b": "Pythia-1B",
    "gemma-3-4b-it": "Gemma-3-4B-IT",
    "gemma-3-12b-it": "Gemma-3-12B-IT",
}

LLM_TO_VLLM_ID_MAPPING = {
    "DeepSeek-7B-Chat": "deepseek-ai/deepseek-llm-7b-chat",
    "Yi-1.5-6B-Chat": "01-ai/Yi-1.5-6B-Chat", 
    "Yi-1.5-9B-Chat-16K": "01-ai/Yi-1.5-9B-Chat-16K",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "MPT-7B-Instruct": "mosaicml/mpt-7b-instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct", 
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Phi-4": "microsoft/phi-4",
    "Falcon3-7B-Instruct": "tiiuae/Falcon3-7B-Instruct",
    "OLMo-2-1B-Instruct": "allenai/OLMo-2-0425-1B-Instruct",
    "OLMo-2-7B-Instruct": "allenai/OLMo-2-1124-7B-Instruct",
    "Pythia-1B": "EleutherAI/pythia-1b",
    "pythia-6.9b-v0": "EleutherAI/pythia-6.9b-v0",
    "Gemma-3-4B-IT": "google/gemma-3-4b-it",
    "Gemma-3-12B-IT": "google/gemma-3-12b-it",
}


def load_vllm_config(config_path: str) -> dict:
    """
    Loads a vLLM configuration file.
    """
    if not config_path.endswith(".yaml") or not os.path.exists(config_path):
        raise ValueError(f"Config path must be a valid YAML file, got {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_vllm_logprobs(raw_logprobs: list[dict[int, Logprob] | None]) -> list[float]:
    logprobs = []
    # the first one is always None, so we skip it
    for logprob in raw_logprobs[1:]:
        if logprob is not None:
            logprobs.append(next(iter(logprob.values())).logprob)
    return logprobs


def set_all_logging_level(level: int, to_stdout: bool = True):
    """
    Sets the logging level for ALL loggers system-wide.
    
    This function:
    1. Configures the root logger (affects all loggers by default)
    2. Handles specific problematic loggers that may have their own handlers
    3. Ensures consistent output to stdout/stderr
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.WARNING)
        to_stdout: If True, output to stdout; otherwise stderr
    """
    import logging
    import sys
    
    # Configure the root logger - this affects all loggers by default
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove all existing handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handler to root logger
    stream = sys.stdout if to_stdout else sys.stderr
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    root_logger.addHandler(handler)
    
    # Handle specific problematic loggers that might have their own configuration
    problematic_loggers = [
        "vllm",
        "vllm.engine", 
        "vllm.worker",
        "vllm.distributed",
        "transformers",
        "torch",
        "pytorch_transformers", 
        "transformers.tokenization_utils",
        "transformers.modeling_utils",
    ]
    
    for logger_name in problematic_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Remove any existing handlers that might override root behavior
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # Don't add handlers here - let them inherit from root
        logger.propagate = True  # Ensure they use root logger's handlers
    
    # Force immediate flush
    stream.flush()