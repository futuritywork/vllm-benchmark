#!/usr/bin/env python3
"""
Core benchmarking functions for vLLM Engine-Direct Connection Ceiling Benchmark
"""

import asyncio
import json
import statistics
import time
import uuid
from typing import List, Optional

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams


def now() -> float:
    """Get current time in seconds"""
    return time.perf_counter()


async def stream_once(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling: SamplingParams,
    tokenizer,
    log_output: bool = False,
    log_file: str = "llm_outputs.log",
) -> dict:
    """
    Start one streamed generation; capture TTFT from the first yielded chunk.
    Keep it alive for `hold_seconds`, then abort to free scheduler/compute.
    """
    rid = str(uuid.uuid4())
    t0 = now()
    first_seen: Optional[float] = None
    ok = False
    error = None
    generated_text = ""
    tokens_generated = 0
    generation_start_time = None
    generation_end_time = None

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

        # Mark as successful if we got any output
        if generated_text:
            ok = True
        else:
            error = "no_generated_text"

    except Exception as e:
        error = repr(e)

    # Ensure cleanup: abort the request if it's still scheduled (only if not successful)
    if not ok:
        try:
            await engine.abort(rid)
        except Exception:
            pass

    # Calculate tokens per second
    total_time = generation_end_time - t0
    tokens_per_second = None

    if tokens_generated > 0:
        tokens_per_second = tokens_generated / total_time

    # Log the output if requested
    if log_output and generated_text:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"=== Request {rid} ===\n")
                f.write(f"Prompt: {prompt[:200]}...\n")
                f.write(f"Generated: {generated_text}\n")
                f.write(f"Total Time: {total_time:.3f}s\n")
                f.write(f"Tokens Generated: {tokens_generated}\n")
                f.write(f"Tokens/Second: {tokens_per_second:.2f} t/s\n")
                f.write(f"Success: {ok and error is None}\n")
                f.write(f"Error: {error}\n")
                f.write("-" * 50 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to log output: {e}")

    return {
        "ok": ok and error is None,
        "total_time": total_time,
        "error": error,
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "tokens_per_second": tokens_per_second,
    }


async def run_level(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling: SamplingParams,
    concurrency: int,
    tokenizer,
    log_output: bool = False,
    log_file: str = "llm_outputs.log",
) -> dict:
    """
    Fire `concurrency` requests simultaneously; aggregate success and TTFT stats.
    """
    print(f"🔥 Firing {concurrency} concurrent requests...")
    tasks = [
        asyncio.create_task(
            stream_once(
                engine,
                prompt,
                sampling,
                tokenizer,
                log_output,
                log_file,
            )
        )
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)

    oks = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    tokens_per_second_list = [
        r["tokens_per_second"] for r in oks if r["tokens_per_second"] is not None
    ]
    tokens_generated_list = [
        r["tokens_generated"] for r in oks if r["tokens_generated"] is not None
    ]

    # Calculate average tokens per second
    avg_tokens_per_second = (
        statistics.mean(tokens_per_second_list) if tokens_per_second_list else 0.0
    )

    # Check if average tokens per second is below threshold
    if avg_tokens_per_second < 25:
        print(
            f"🚨 PERFORMANCE THRESHOLD EXCEEDED: Average tokens/second ({avg_tokens_per_second:.2f}) is below 25 t/s threshold!"
        )
        print(f"   This concurrency level ({concurrency}) is not sustainable.")
        # Mark as failed if performance is too low
        oks = []
        fails = results
        ok_rate = 0.0
    else:
        ok_rate = (len(oks) / len(results)) if results else 0.0

    summary = {
        "concurrency": concurrency,
        "total": len(results),
        "ok": len(oks),
        "fail": len(fails),
        "ok_rate": ok_rate,
        "avg_tokens_per_second": avg_tokens_per_second,
        "tokens_per_second_p50": (
            statistics.median(tokens_per_second_list)
            if tokens_per_second_list
            else None
        ),
        "tokens_per_second_p95": (
            (
                sorted(tokens_per_second_list)[
                    max(0, int(0.95 * (len(tokens_per_second_list) - 1)))
                ]
            )
            if tokens_per_second_list
            else None
        ),
        "avg_tokens_generated": (
            statistics.mean(tokens_generated_list) if tokens_generated_list else None
        ),
        "errors_sample": fails[:5],
    }
    return summary


async def find_ceiling(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling: SamplingParams,
    start_conc: int,
    max_conc_cap: int,
    sla_ok_rate: float,
    tokenizer,
    log_output: bool = False,
    log_file: str = "llm_outputs.log",
) -> dict:
    """
    Exponential ramp to first failure, then binary-search to find the max concurrency
    where ok_rate ≥ sla_ok_rate.
    """
    history: List[dict] = []
    conc = max(1, start_conc)
    last_pass = 0
    first_fail = None

    # Exponential ramp
    print(f"\n🚀 Starting exponential ramp from {start_conc} to {max_conc_cap}...")
    while conc <= max_conc_cap:
        print(f"\n📊 Testing concurrency level: {conc}")
        res = await run_level(
            engine,
            prompt,
            sampling,
            conc,
            tokenizer,
            log_output,
            log_file,
        )
        history.append(res)
        print(f"[ramp] {json.dumps(res)}")

        if res["ok_rate"] >= sla_ok_rate:
            print(f"✅ Concurrency {conc} passed (success rate: {res['ok_rate']:.1%})")
            last_pass = conc
            conc *= 2
        else:
            print(f"❌ Concurrency {conc} failed (success rate: {res['ok_rate']:.1%})")
            first_fail = conc
            break

    if first_fail is None:
        # Never failed before cap
        print(
            f"🎉 Never hit failure threshold - max sustainable concurrency: {last_pass}"
        )
        return {"max_sustainable": last_pass, "history": history}

    # Binary search (last_pass+1 .. first_fail-1)
    print(
        f"\n🔍 Starting binary search between {last_pass + 1} and {first_fail - 1}..."
    )
    lo = last_pass + 1
    hi = first_fail - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        print(f"\n📊 Testing concurrency level: {mid} (range: {lo}-{hi})")
        res = await run_level(
            engine,
            prompt,
            sampling,
            mid,
            tokenizer,
            log_output,
            log_file,
        )
        history.append(res)
        print(f"[search] {json.dumps(res)}")
        if res["ok_rate"] >= sla_ok_rate:
            print(f"✅ Concurrency {mid} passed (success rate: {res['ok_rate']:.1%})")
            last_pass = mid
            lo = mid + 1
        else:
            print(f"❌ Concurrency {mid} failed (success rate: {res['ok_rate']:.1%})")
            hi = mid - 1

    print(f"\n🎯 Binary search complete! Max sustainable concurrency: {last_pass}")
    return {"max_sustainable": last_pass, "history": history}
