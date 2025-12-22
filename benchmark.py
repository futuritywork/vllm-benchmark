#!/usr/bin/env python3
"""
vLLM Concurrent Benchmark Script

This script benchmarks vLLM by running concurrent inference requests with:
1. Exponential search (2^n) to find bounds
2. Binary search to find maximum concurrent requests maintaining ≥25 tokens/sec average
"""

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from vllm import LLM, SamplingParams


def get_available_gpus() -> int:
    """Detect the number of available GPUs."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: check CUDA_VISIBLE_DEVICES environment variable
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Count comma-separated GPU indices
        return len([x for x in cuda_visible.split(",") if x.strip()])

    return 1  # Default to 1 if we can't detect


class BenchmarkResult:
    """Stores results for a single benchmark run."""

    def __init__(
        self,
        concurrency: int,
        total_time: float,
        total_tokens: int,
        tokens_per_second: float,
        success: bool = True,
        error: str = None,
    ):
        self.concurrency = concurrency
        self.total_time = total_time
        self.total_tokens = total_tokens
        self.tokens_per_second = tokens_per_second
        self.success = success
        self.error = error

    def to_dict(self) -> Dict:
        return {
            "concurrency": self.concurrency,
            "total_time": self.total_time,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
            "success": self.success,
            "error": self.error,
        }


async def run_concurrent_requests(
    llm: LLM,
    prompt: str,
    num_requests: int,
    max_tokens: int = 10000,
    timeout: float = 600.0,  # 10 minutes default timeout
) -> Tuple[float, int, bool, str]:
    """
    Run multiple concurrent inference requests and measure performance.

    Returns:
        (total_time, total_tokens, success, error_message)
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    async def single_request():
        """Run a single inference request."""
        try:
            # Run generation in thread pool to avoid blocking
            # Add timeout to prevent hanging requests
            outputs = await asyncio.wait_for(
                asyncio.to_thread(llm.generate, [prompt], sampling_params),
                timeout=timeout,
            )
            if not outputs or len(outputs) == 0:
                return (0, False, "Empty output")
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            return (tokens_generated, True, None)
        except asyncio.TimeoutError:
            return (0, False, f"Request timeout after {timeout}s")
        except RuntimeError as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or "oom" in error_str:
                return (0, False, "Out of memory (OOM)")
            return (0, False, f"Runtime error: {e}")
        except Exception as e:
            error_str = str(e).lower()
            # Categorize common errors
            if "cuda" in error_str or "gpu" in error_str:
                return (0, False, f"GPU error: {e}")
            return (0, False, f"Error: {e}")

    # Create all tasks
    tasks = [single_request() for _ in range(num_requests)]

    # Run all requests concurrently
    start_time = time.perf_counter()
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        # Catastrophic failure
        return (0.0, 0, False, f"Failed to run requests: {e}")
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_tokens = 0
    all_success = True
    errors = []
    error_counts = {}

    for result in results:
        if isinstance(result, Exception):
            all_success = False
            error_msg = str(result)
            errors.append(error_msg)
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
        else:
            tokens, success, error = result
            if not success:
                all_success = False
                if error:
                    errors.append(error)
                    error_counts[error] = error_counts.get(error, 0) + 1
            total_tokens += tokens

    # Create concise error message
    if error_counts:
        error_parts = [f"{count}x {err}" for err, count in error_counts.items()]
        error_message = "; ".join(error_parts)
    else:
        error_message = None

    return (total_time, total_tokens, all_success, error_message)


async def exponential_search(
    llm: LLM,
    prompt: str,
    min_tokens_per_second: float = 25.0,
    max_concurrency: int = 1024,
    timeout: float = 600.0,
    max_tokens: int = 10000,
) -> Tuple[List[BenchmarkResult], int, int]:
    """
    Perform exponential search (2^n) to find bounds.

    Returns:
        (results, lower_bound, upper_bound)
        lower_bound: last concurrency level with tokens/sec >= min_tokens_per_second
        upper_bound: first concurrency level with tokens/sec < min_tokens_per_second
    """
    results = []
    n = 0
    lower_bound = None
    upper_bound = None

    print("Starting exponential search phase...")
    print(f"Target: ≥{min_tokens_per_second} tokens/sec average")
    print(f"Maximum concurrency limit: {max_concurrency}")
    print("-" * 60)

    while True:
        concurrency = 2**n

        # Safety check to prevent excessive concurrency
        if concurrency > max_concurrency:
            print(f"Reached maximum concurrency limit: {max_concurrency}")
            if lower_bound is None:
                print(
                    "Warning: No concurrency level met the target before hitting limit!"
                )
            upper_bound = concurrency
            break

        print(f"Testing concurrency level: {concurrency} (2^{n})")

        try:
            total_time, total_tokens, success, error = await run_concurrent_requests(
                llm, prompt, concurrency, max_tokens=max_tokens, timeout=timeout
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            raise
        except Exception as e:
            print(f"  Fatal error: {e}")
            upper_bound = concurrency
            result = BenchmarkResult(
                concurrency=concurrency,
                total_time=0.0,
                total_tokens=0,
                tokens_per_second=0.0,
                success=False,
                error=f"Fatal error: {e}",
            )
            results.append(result)
            break

        if total_time > 0 and total_tokens > 0:
            tokens_per_second = total_tokens / total_time
        else:
            tokens_per_second = 0.0

        result = BenchmarkResult(
            concurrency=concurrency,
            total_time=total_time,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            success=success,
            error=error,
        )
        results.append(result)

        print(
            f"  Time: {total_time:.2f}s, Tokens: {total_tokens}, "
            f"Tokens/sec: {tokens_per_second:.2f}"
        )

        if not success:
            print(f"  Error: {error}")
            upper_bound = concurrency
            break

        if tokens_per_second >= min_tokens_per_second:
            lower_bound = concurrency
            print(f"  ✓ Meets target (≥{min_tokens_per_second} tokens/sec)")
        else:
            upper_bound = concurrency
            print(f"  ✗ Below target (<{min_tokens_per_second} tokens/sec)")
            break

        n += 1

    print("-" * 60)
    if lower_bound is None:
        print("Warning: No concurrency level met the target!")
    elif upper_bound is None:
        print("Warning: Upper bound not found, performance may degrade further.")
    else:
        print(f"Bounds found: lower={lower_bound}, upper={upper_bound}")

    return results, lower_bound, upper_bound


async def binary_search(
    llm: LLM,
    prompt: str,
    lower_bound: int,
    upper_bound: int,
    min_tokens_per_second: float = 25.0,
    timeout: float = 600.0,
    max_tokens: int = 10000,
) -> List[BenchmarkResult]:
    """
    Perform binary search between bounds to find maximum concurrent requests
    maintaining ≥min_tokens_per_second average.

    Returns:
        List of benchmark results from binary search
    """
    results = []

    if lower_bound is None or upper_bound is None:
        print("Skipping binary search: bounds not found")
        return results

    print("\nStarting binary search phase...")
    print(f"Searching between {lower_bound} and {upper_bound} concurrent requests")
    print(f"Target: ≥{min_tokens_per_second} tokens/sec average")
    print("-" * 60)

    left = lower_bound
    right = upper_bound
    best = lower_bound

    while left < right - 1:
        mid = (left + right) // 2
        print(f"Testing concurrency level: {mid}")

        try:
            total_time, total_tokens, success, error = await run_concurrent_requests(
                llm, prompt, mid, max_tokens=max_tokens, timeout=timeout
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            raise
        except Exception as e:
            print(f"  Fatal error: {e}")
            total_time = 0.0
            total_tokens = 0
            success = False
            error = f"Fatal error: {e}"

        if total_time > 0 and total_tokens > 0:
            tokens_per_second = total_tokens / total_time
        else:
            tokens_per_second = 0.0

        result = BenchmarkResult(
            concurrency=mid,
            total_time=total_time,
            total_tokens=total_tokens,
            tokens_per_second=tokens_per_second,
            success=success,
            error=error,
        )
        results.append(result)

        print(
            f"  Time: {total_time:.2f}s, Tokens: {total_tokens}, "
            f"Tokens/sec: {tokens_per_second:.2f}"
        )

        if success and tokens_per_second >= min_tokens_per_second:
            best = mid
            left = mid
            print(f"  ✓ Meets target, searching higher")
        else:
            right = mid
            if not success:
                print(f"  ✗ Error: {error}")
            else:
                print(f"  ✗ Below target, searching lower")

    print("-" * 60)
    print(f"Maximum concurrent requests: {best}")

    return results


def save_results(
    exponential_results: List[BenchmarkResult],
    binary_results: List[BenchmarkResult],
    output_dir: Path,
):
    """Save results in JSON, CSV, and print human-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = exponential_results + binary_results

    # JSON output
    json_path = output_dir / "benchmark_results.json"
    json_data = {
        "exponential_search": [r.to_dict() for r in exponential_results],
        "binary_search": [r.to_dict() for r in binary_results],
        "all_results": [r.to_dict() for r in all_results],
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # CSV output
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "concurrency",
                "total_time",
                "total_tokens",
                "tokens_per_second",
                "success",
                "error",
                "phase",
            ]
        )
        for r in exponential_results:
            writer.writerow(
                [
                    r.concurrency,
                    r.total_time,
                    r.total_tokens,
                    r.tokens_per_second,
                    r.success,
                    r.error or "",
                    "exponential",
                ]
            )
        for r in binary_results:
            writer.writerow(
                [
                    r.concurrency,
                    r.total_time,
                    r.total_tokens,
                    r.tokens_per_second,
                    r.success,
                    r.error or "",
                    "binary",
                ]
            )
    print(f"CSV results saved to: {csv_path}")

    # Human-readable summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print("\nExponential Search Results:")
    print(
        f"{'Concurrency':<12} {'Time (s)':<12} {'Tokens':<12} {'Tokens/s':<12} {'Status':<10}"
    )
    print("-" * 60)
    for r in exponential_results:
        status = "✓ PASS" if r.success and r.tokens_per_second >= 25.0 else "✗ FAIL"
        print(
            f"{r.concurrency:<12} {r.total_time:<12.2f} {r.total_tokens:<12} "
            f"{r.tokens_per_second:<12.2f} {status:<10}"
        )

    if binary_results:
        print("\nBinary Search Results:")
        print(
            f"{'Concurrency':<12} {'Time (s)':<12} {'Tokens':<12} {'Tokens/s':<12} {'Status':<10}"
        )
        print("-" * 60)
        for r in binary_results:
            status = "✓ PASS" if r.success and r.tokens_per_second >= 25.0 else "✗ FAIL"
            print(
                f"{r.concurrency:<12} {r.total_time:<12.2f} {r.total_tokens:<12} "
                f"{r.tokens_per_second:<12.2f} {status:<10}"
            )

    # Find best result
    passing_results = [
        r for r in all_results if r.success and r.tokens_per_second >= 25.0
    ]
    if passing_results:
        best = max(passing_results, key=lambda r: r.concurrency)
        print(f"\nBest Result: {best.concurrency} concurrent requests")
        print(f"  Tokens/sec: {best.tokens_per_second:.2f}")
        print(f"  Total tokens: {best.total_tokens}")
        print(f"  Total time: {best.total_time:.2f}s")
    else:
        print("\nNo results met the target of ≥25 tokens/sec")
    print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM with concurrent inference requests"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., mistralai/Mistral-7B-Instruct-v0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="1984_prompt.txt",
        help="Path to prompt file (default: 1984_prompt.txt)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate per request (default: 10000)",
    )
    parser.add_argument(
        "--min-tokens-per-second",
        type=float,
        default=25.0,
        help="Minimum tokens per second target (default: 25.0)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1024,
        help="Maximum concurrency limit for safety (default: 1024)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout per request in seconds (default: 600.0)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism (default: auto-detect all available GPUs)",
    )

    args = parser.parse_args()

    # Load prompt
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}")
        return 1

    with open(prompt_path, "r") as f:
        prompt = f.read()

    print(f"Loaded prompt from: {prompt_path}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Model: {args.model}")
    print(f"Max tokens per request: {args.max_tokens}")

    # Detect available GPUs
    available_gpus = get_available_gpus()
    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = available_gpus
        print(f"Detected {available_gpus} GPU(s), using all of them")
    else:
        if tensor_parallel_size > available_gpus:
            print(
                f"Warning: Requested {tensor_parallel_size} GPUs but only {available_gpus} available"
            )
            tensor_parallel_size = available_gpus
        print(
            f"Using {tensor_parallel_size} GPU(s) (out of {available_gpus} available)"
        )
    print()

    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(model=args.model, tensor_parallel_size=tensor_parallel_size)
    print("vLLM initialized")
    print()

    try:
        # Exponential search phase
        exponential_results, lower_bound, upper_bound = await exponential_search(
            llm,
            prompt,
            args.min_tokens_per_second,
            args.max_concurrency,
            args.timeout,
            args.max_tokens,
        )

        # Binary search phase
        binary_results = await binary_search(
            llm,
            prompt,
            lower_bound,
            upper_bound,
            args.min_tokens_per_second,
            args.timeout,
            args.max_tokens,
        )

        # Save results
        output_dir = Path(args.output_dir)
        save_results(exponential_results, binary_results, output_dir)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
