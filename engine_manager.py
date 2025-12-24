#!/usr/bin/env python3
"""
Engine management utilities for vLLM Engine-Direct Connection Ceiling Benchmark
"""

import os

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

from config import BenchmarkConfig


def create_engine(config: BenchmarkConfig) -> AsyncLLMEngine:
    """
    Create and initialize AsyncLLMEngine with the given configuration.
    """
    # Set environment variable for YARN if needed
    if config.allow_long_max_model_len:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    engine_kwargs = {
        "model": config.model,
        "dtype": config.dtype,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "swap_space": config.swap_space,
        "enforce_eager": config.enforce_eager,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = config.max_num_seqs
    if config.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
    if config.rope_scaling is not None:
        engine_kwargs["rope_scaling"] = config.rope_scaling

    engine_args = AsyncEngineArgs(**engine_kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args)


def create_sampling_params(config: BenchmarkConfig) -> SamplingParams:
    """
    Create sampling parameters for the benchmark.
    """
    return SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=0.7,  # Add randomness for variety
        top_p=0.9,  # Nucleus sampling for better quality
        top_k=50,  # Limit to top 50 tokens for diversity
    )
