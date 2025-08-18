#!/usr/bin/env python3
"""
Prompt construction utilities for vLLM Engine-Direct Connection Ceiling Benchmark
"""

from typing import Optional, Tuple

# Optional (token counting); handled gracefully if unavailable
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


async def build_prompt_of_tokens(
    model_or_tokenizer_id: str,
    target_tokens: int,
    seed_chunk: str,
    trust_remote_code: bool = False,
) -> Tuple[str, Optional[int]]:
    """
    Try to construct a prompt with ~target_tokens using a HF tokenizer for accuracy.
    Fallback: ~4 chars per token approximation if tokenizer isn't available.
    Returns (prompt_text, measured_token_count_or_None).
    """
    # 1) Start with a rough character-based expansion.
    approx_chars_per_token = 4
    target_chars = target_tokens * approx_chars_per_token
    text = seed_chunk
    while len(text) < target_chars:
        text += seed_chunk

    # 2) If we can, calibrate with a tokenizer (prefer exactness for 5k).
    if AutoTokenizer is None:
        return text, None

    try:
        tok = AutoTokenizer.from_pretrained(
            model_or_tokenizer_id, trust_remote_code=trust_remote_code, use_fast=True
        )
        # Grow or shrink to reach/just exceed target
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < target_tokens:
            while len(ids) < target_tokens:
                text += seed_chunk
                ids = tok.encode(text, add_special_tokens=False)
        # Optional light trim if grossly oversized (not strictly necessary)
        return text, len(ids)
    except Exception:
        return text, None
