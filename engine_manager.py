#!/usr/bin/env python3
"""
Engine management utilities for vLLM Engine-Direct Connection Ceiling Benchmark
"""

import sys
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

from config import BenchmarkConfig


def create_engine(config: BenchmarkConfig) -> AsyncLLMEngine:
    """
    Create and initialize AsyncLLMEngine with the given configuration.
    """
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

    try:
        engine_args = AsyncEngineArgs(**engine_kwargs)
        return AsyncLLMEngine.from_engine_args(engine_args)
    except (TypeError, RuntimeError, Exception) as e:
        error_msg = str(e)
        error_repr = repr(e)
        # Check for the specific processor type error (can appear in error message or traceback)
        if (
            "ProcessorMixin" in error_msg
            or "PreTrainedTokenizerFast" in error_msg
            or "Invalid type of HuggingFace processor" in error_msg
            or "ProcessorMixin" in error_repr
            or "PreTrainedTokenizerFast" in error_repr
        ):
            print("\n" + "=" * 80, file=sys.stderr)
            print("ERROR: vLLM processor type mismatch detected", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(
                f"\nThe model '{config.model}' appears to be a multimodal model,",
                file=sys.stderr,
            )
            print(
                "but vLLM is encountering a processor type mismatch during initialization.",
                file=sys.stderr,
            )
            print(
                "\nThis is a known compatibility issue with some multimodal models in vLLM v1.",
                file=sys.stderr,
            )
            print(
                "The error occurs when vLLM tries to load a processor for multimodal profiling,",
                file=sys.stderr,
            )
            print(
                "but receives a tokenizer instead of the expected ProcessorMixin.",
                file=sys.stderr,
            )
            print("\nPossible solutions:", file=sys.stderr)
            print(
                "1. Check if the model is compatible with your vLLM version",
                file=sys.stderr,
            )
            print("2. Try updating vLLM to the latest version:", file=sys.stderr)
            print("   pip install --upgrade vllm", file=sys.stderr)
            print(
                "3. Use a different model that is known to work with vLLM",
                file=sys.stderr,
            )
            print(
                "4. Check vLLM GitHub issues for this specific model:", file=sys.stderr
            )
            print(
                f"   https://github.com/vllm-project/vllm/issues?q={config.model}",
                file=sys.stderr,
            )
            print("\nOriginal error:", file=sys.stderr)
            print("-" * 80, file=sys.stderr)
        raise


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
