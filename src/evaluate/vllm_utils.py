"""
Utility functions for vLLM inference.
"""
import yaml
import os
from vllm.sequence import Logprob


def load_vllm_config(config_path: str) -> dict:
    """
    Loads a vLLM configuration file.
    """
    if not config_path.endswith(".yaml") or not os.path.exists(config_path):
        raise ValueError(f"Config path must be a valid YAML file, got {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_logprobs(raw_logprobs: list[dict[int, Logprob] | None]) -> list[float]:
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