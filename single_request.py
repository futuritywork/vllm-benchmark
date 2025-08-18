#!/usr/bin/env python3
"""
Single Request Token Speed Test

This script runs a single LLM request and measures the tokens per second.
"""

import asyncio
import time
import argparse
from typing import Optional

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


def now() -> float:
    """Get current time in seconds"""
    return time.perf_counter()


async def run_single_request(
    model: str,
    prompt: str,
    max_new_tokens: int = 500,
    gpu_memory_utilization: float = 0.7,
    max_num_seqs: int = 64,
    log_file: str = "single_request.log",
) -> dict:
    """
    Run a single LLM request and measure performance.
    """
    print(f"Loading model: {model}")

    # Create engine
    engine_args = AsyncEngineArgs(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Create sampling params
    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.7,  # Add randomness for variety
        top_p=0.9,  # Nucleus sampling for better quality
        top_k=50,  # Limit to top 50 tokens for diversity
    )

    print(f"Running request with {max_new_tokens} max tokens...")
    print(f"Prompt length: {len(prompt)} characters")

    # Run the request
    rid = "single_request_test"
    t0 = now()
    first_seen: Optional[float] = None
    tokens_generated = 0
    generation_start_time = None
    generation_end_time = None
    generated_text = ""

    try:
        # Let the engine run to completion and collect all outputs
        outputs = []
        async for out in engine.generate(prompt, sampling, request_id=rid):
            outputs.append(out)

        # Get the final output
        if outputs and len(outputs) > 0:
            final_output = outputs[-1]  # Get the last output
            if hasattr(final_output, "outputs") and final_output.outputs:
                output = final_output.outputs[0]

                # Get the generated text
                if hasattr(output, "text"):
                    full_text = output.text
                    if full_text.startswith(prompt):
                        generated_text = full_text[len(prompt) :]
                    else:
                        generated_text = full_text

                # Get timing info if available
                if hasattr(output, "finish_reason"):
                    print(f"Finish reason: {output.finish_reason}")

        # Record end time
        generation_end_time = now()

        # Count tokens at the end using the tokenizer
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

            # Count tokens in the generated text only
            if generated_text:
                tokens_generated = len(
                    tokenizer.encode(generated_text, add_special_tokens=False)
                )
            else:
                tokens_generated = 0

        except Exception as e:
            print(f"Warning: Could not count tokens with tokenizer: {e}")
            tokens_generated = 0

    except Exception as e:
        print(f"Error during generation: {e}")
        return {"error": str(e)}

    # Calculate metrics
    total_time = generation_end_time - t0
    tokens_per_second = None

    if tokens_generated > 0:
        tokens_per_second = tokens_generated / total_time

    # Log results
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Single Request Test ===\n")
            f.write(f"Model: {model}\n")
            f.write(f"Max Tokens: {max_new_tokens}\n")
            f.write(f"Prompt Length: {len(prompt)} characters\n")
            f.write(f"Generated Text: {generated_text}\n")
            f.write(f"Tokens Generated: {tokens_generated}\n")
            f.write(f"Total Time: {total_time:.3f}s\n")
            f.write(f"Tokens/Second: {tokens_per_second:.2f} t/s\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"Warning: Failed to log output: {e}")

    # Print results to console
    print(f"\n=== Results ===")
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Total Time: {total_time:.3f}s")
    print(f"Tokens/Second: {tokens_per_second:.2f} t/s")
    print(f"Results logged to: {log_file}")

    return {
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
        "generated_text": generated_text,
        "total_time": total_time,
    }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Single request token speed test")
    parser.add_argument("--model", required=True, help="Model to test")
    parser.add_argument(
        "--prompt-file", default="conversation.txt", help="File containing prompt"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=500, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=64, help="Maximum number of sequences"
    )
    parser.add_argument(
        "--log-file", default="single_request.log", help="Output log file"
    )

    args = parser.parse_args()

    # Read prompt from file
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{args.prompt_file}' not found")
        return
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # Run the test
    result = await run_single_request(
        model=args.model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        log_file=args.log_file,
    )

    if "error" in result:
        print(f"Test failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
