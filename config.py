#!/usr/bin/env python3
"""
Configuration and argument parsing for vLLM Engine-Direct Connection Ceiling Benchmark
"""

import argparse
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run"""

    # Benchmark behavior
    target_input_tokens: int
    max_new_tokens: int
    start_concurrency: int
    max_concurrency_cap: int
    sla_ok_rate: float
    json_out: str
    tokenizer: str | None
    trust_remote_code: bool
    log_output: bool
    log_file: str

    # Engine args
    model: str
    dtype: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    swap_space: int
    enforce_eager: bool


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments and return configuration"""
    p = argparse.ArgumentParser(
        description="vLLM engine-direct concurrency ceiling (≈5k ctx)"
    )

    # Benchmark behavior
    p.add_argument("--target-input-tokens", type=int, default=5000)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--start-concurrency", type=int, default=1)
    p.add_argument("--max-concurrency-cap", type=int, default=4096)
    p.add_argument("--sla-ok-rate", type=float, default=0.99)
    p.add_argument("--json-out", default="engine_conn_results.json")
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer ID to measure tokens (defaults to --model)",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--log-output", action="store_true", help="Log LLM outputs to file")
    p.add_argument(
        "--log-file", default="llm_outputs.log", help="File to log LLM outputs"
    )

    # Engine args (common subset; mirrors vLLM CLI flags)
    p.add_argument("--model", required=True, help="HF or local model path")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Engine scheduler limit (optional)",
    )
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="CPU swap space (GiB) for paged KV if enabled",
    )
    p.add_argument(
        "--enforce-eager", action="store_true", help="Disable CUDA graph capture if set"
    )

    args = p.parse_args()

    return BenchmarkConfig(
        target_input_tokens=args.target_input_tokens,
        max_new_tokens=args.max_new_tokens,
        start_concurrency=args.start_concurrency,
        max_concurrency_cap=args.max_concurrency_cap,
        sla_ok_rate=args.sla_ok_rate,
        json_out=args.json_out,
        tokenizer=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        log_output=args.log_output,
        log_file=args.log_file,
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        swap_space=args.swap_space,
        enforce_eager=args.enforce_eager,
    )
