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
import json
import os

from transformers import AutoTokenizer

from config import parse_args
from prompt_builder import build_prompt_of_tokens
from engine_manager import create_engine, create_sampling_params
from benchmark import run_level, find_ceiling


async def main():
    """Main entry point for the benchmark"""
    config = parse_args()

    # 1) Build ~5k-token prompt
    tokenizer_id = config.tokenizer or config.model

    f = open("1984.txt", "r")
    prompt = f.read()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    print(
        f"[prompt] target={config.target_input_tokens} measured={len(ids)} chars={len(prompt)}"
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

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK RESULTS")
    print("=" * 60)

    max_sustainable = result["max_sustainable"]
    history = result.get("history", [])

    print(f"📊 Max Sustainable Concurrency: {max_sustainable}")
    print(f"🎯 Success Rate Threshold: ≥ {config.sla_ok_rate:.1%}")
    print(f"⚡ Performance Threshold: ≥ 25 tokens/second")
    print(f"🕐 Timeout: ≤ {config.ttft_timeout}s")
    print(f"🔢 Max Tokens per Request: {config.max_new_tokens}")
    # Show CUDA_VISIBLE_DEVICES if set
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices:
        print(f"🖥️  CUDA_VISIBLE_DEVICES: {cuda_devices}")

    if history:
        print(f"\n📈 Performance Summary:")
        print(
            f"{'Concurrency':<12} {'Success Rate':<12} {'Avg Tokens/s':<12} {'P50 Tokens/s':<12} {'P95 Tokens/s':<12}"
        )
        print("-" * 60)

        for res in history:
            conc = res.get("concurrency", 0)
            ok_rate = res.get("ok_rate", 0)
            avg_tps = res.get("avg_tokens_per_second", 0)
            p50_tps = res.get("tokens_per_second_p50", 0)
            p95_tps = res.get("tokens_per_second_p95", 0)

            status = "✅" if ok_rate >= config.sla_ok_rate else "❌"
            print(
                f"{conc:<12} {ok_rate:.1%} {status:<2} {avg_tps:<12.1f} {p50_tps:<12.1f} {p95_tps:<12.1f}"
            )

    print(f"\n💾 Results saved to: {config.json_out}")
    if config.log_output:
        print(f"📝 Detailed logs saved to: {config.log_file}")

    print("=" * 60)

    # Save detailed results to JSON
    with open(config.json_out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
