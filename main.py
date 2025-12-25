#!/usr/bin/env python3
"""
vLLM Engine-Direct Connection Ceiling Benchmark (≈5k-token prefill)

What it measures
----------------
Find the highest in-flight concurrency (N) that the *engine* can sustain
for a long prompt (default ~5k tokens) while meeting an SLA on:
  • Success rate (opened stream and produced first token)
  • Time-to-first-token (TTFT) threshold

This avoids HTTP/gRPC entirely by driving vLLM's AsyncLLMEngine directly.

How it works
------------
1) Build a ~target-token prompt (tries HF tokenizer; falls back to char-length).
2) Warm up with a single request.
3) Ramp concurrency: 1→2→4→… until SLA fails; then binary-search the ceiling.
4) Each request records TTFT at the first yielded chunk; then keeps generating
   for `hold_seconds` to keep the sequence "in flight", and aborts cleanly.

Usage
-----
python3 main.py \
  --model meta-llama/Llama-3.1-8B \
  --target-input-tokens 5000 \
  --ttft-timeout 60 \
  --hold-seconds 5 \
  --sla-ok-rate 0.99

Common engine args (mirrors CLI):
  --dtype auto|float16|bfloat16
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.9
  --max-model-len 8192
  --max-num-seqs 2048

Note: GPU selection is controlled via CUDA_VISIBLE_DEVICES environment variable
"""

import asyncio
from datetime import datetime
import json
import os
import time

from transformers import AutoTokenizer

from config import parse_args
from engine_manager import create_engine, create_sampling_params
from benchmark import run_level, find_ceiling
from prompt_selector import get_prompt_path
from random_prompt_generator import generate_random_prompt, prompt_for_token_count


async def main():
    """Main entry point for the benchmark"""
    benchmark_start_time = datetime.now()
    benchmark_start_timestamp = benchmark_start_time.strftime("%Y-%m-%d %H:%M:%S")
    benchmark_start_perf = time.perf_counter()
    
    print("=" * 60)
    print("🚀 BENCHMARK STARTED")
    print(f"⏰ Start time: {benchmark_start_timestamp}")
    print("=" * 60)
    
    config = parse_args()

    # 1) Load tokenizer first (needed for both modes)
    tokenizer_id = config.tokenizer or config.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )

    # 2) Get or generate prompt based on mode
    if config.random_tokens:
        # Random token mode: prompt for token count and generate random prompt
        target_tokens = prompt_for_token_count()
        print(f"\n🎲 Generating random prompt with ~{target_tokens:,} tokens...")
        prompt, prompt_tokens = generate_random_prompt(target_tokens, tokenizer)
        print(
            f"[random prompt] target={target_tokens:,} measured={prompt_tokens:,} chars={len(prompt):,}"
        )
        error_pct = abs(prompt_tokens - target_tokens) / target_tokens * 100
        print(f"[random prompt] error={error_pct:.2f}%")
    else:
        # File-based mode: select prompt file (interactive if not provided)
        prompt_path = get_prompt_path(config.prompt)
        print(f"📄 Loading prompt from: {prompt_path}")
        
        with open(prompt_path, "r") as f:
            prompt = f.read()

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_tokens = len(ids)
        print(
            f"[prompt] target={config.target_input_tokens} measured={prompt_tokens} chars={len(prompt)}"
        )

    # 2) Spin up AsyncLLMEngine
    engine = create_engine(config)

    # 3) Sampling params: tiny decode to keep streams alive
    sampling = create_sampling_params(config)

    # 4) Warm-up (helps avoid first-iteration compilation/graph-capture skew)
    warm = await run_level(
        engine,
        prompt,
        sampling,
        concurrency=1,
        tokenizer=tokenizer,
        log_output=config.log_output,
        log_file=config.log_file,
    )
    print(f"[warmup] {warm}")

    # 5) Ramp + binary search to find ceiling
    result = await find_ceiling(
        engine=engine,
        prompt=prompt,
        sampling=sampling,
        start_conc=config.start_concurrency,
        max_conc_cap=config.max_concurrency_cap,
        sla_ok_rate=config.sla_ok_rate,
        tokenizer=tokenizer,
        log_output=config.log_output,
        log_file=config.log_file,
    )
    
    benchmark_end_perf = time.perf_counter()
    benchmark_end_time = datetime.now()
    benchmark_end_timestamp = benchmark_end_time.strftime("%Y-%m-%d %H:%M:%S")
    benchmark_duration = benchmark_end_perf - benchmark_start_perf

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"⏰ Benchmark started at: {benchmark_start_timestamp}")
    print(f"⏰ Benchmark ended at: {benchmark_end_timestamp}")
    print(f"⏱️  Total benchmark duration: {benchmark_duration:.2f} seconds ({benchmark_duration/60:.2f} minutes)")
    print("=" * 60)

    max_sustainable = result.max_sustainable
    history = result.history

    print(f"📊 Max Sustainable Concurrency: {max_sustainable}")
    print(f"🎯 Success Rate Threshold: ≥ {config.sla_ok_rate:.1%}")
    print(f"⚡ Performance Threshold: ≥ 25 tokens/second")
    print(f"🔢 Max Tokens per Request: {config.max_new_tokens}")
    print(f"📝 Prompt Length: {prompt_tokens} tokens")
    # Show CUDA_VISIBLE_DEVICES if set
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices:
        print(f"🖥️  CUDA_VISIBLE_DEVICES: {cuda_devices}")
    # Show YARN configuration if set
    if config.rope_scaling:
        print(f"🧵 RoPE Scaling: {json.dumps(config.rope_scaling)}")
    if config.allow_long_max_model_len:
        print(f"🔓 VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1")

    if history:
        print(f"\n📈 Performance Summary:")
        print(
            f"{'Concurrency':<12} {'Success Rate':<12} {'Avg Tokens/s':<12} {'P50 Tokens/s':<12} {'P95 Tokens/s':<12} {'Total Tokens':<14} {'Total TPS':<12}"
        )
        print("-" * 60)

        for res in history:
            conc = res.concurrency
            ok_rate = res.ok_rate
            avg_tps = res.avg_tokens_per_second
            p50_tps = res.tokens_per_second_p50
            p95_tps = res.tokens_per_second_p95
            total_tokens = res.total_tokens_generated
            total_tps = res.total_tokens_per_second

            status = "✅" if ok_rate >= config.sla_ok_rate else "❌"
            print(
                f"{conc:<12} {ok_rate:.1%} {status:<2} {avg_tps:<12.1f} {p50_tps:<12.1f} {p95_tps:<12.1f} {total_tokens:<14} {total_tps:<12.1f}"
            )

    print(f"\n💾 Results saved to: {config.json_out}")
    if config.log_output:
        print(f"📝 Detailed logs saved to: {config.log_file}")

    print("=" * 60)

    # Save detailed results to JSON
    with open(config.json_out, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
