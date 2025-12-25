#!/usr/bin/env python3
"""
Random prompt generator for vLLM benchmark
Generates prompts with random English words to achieve a target token count
"""

import os
import random
from typing import Tuple

# Error tolerance for token count (5%)
TOKEN_ERROR_TOLERANCE = 0.05

# Target ratio of words to tokens (words are typically ~0.75-1.3 tokens on average)
WORDS_TO_TOKENS_RATIO = 0.85

# Word corpus file path (Google 10000 English words, no swears)
# Source: https://github.com/first20hours/google-10000-english
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "google-10000-english-usa-no-swears.txt")


def load_corpus(max_words: int = 1000) -> list[str]:
    """
    Load the English word corpus from file.
    
    Args:
        max_words: Maximum number of words to load (most frequent first)
        
    Returns:
        List of words
    """
    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_FILE}\n"
            "Download from: https://github.com/first20hours/google-10000-english"
        )
    
    with open(CORPUS_FILE, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    
    return words[:max_words]


# Lazy-loaded corpus (loaded on first use)
_corpus_cache: list[str] | None = None


def get_corpus() -> list[str]:
    """Get the word corpus, loading it if necessary."""
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = load_corpus()
    return _corpus_cache


def generate_random_prompt(
    target_tokens: int,
    tokenizer,
    max_iterations: int = 50,
) -> Tuple[str, int]:
    """
    Generate a random prompt with approximately the target number of tokens.
    
    Uses random English words and adjusts until within TOKEN_ERROR_TOLERANCE (5%).
    
    Args:
        target_tokens: Target number of tokens
        tokenizer: HuggingFace tokenizer to count tokens
        max_iterations: Maximum adjustment iterations
        
    Returns:
        Tuple of (generated_prompt, actual_token_count)
    """
    corpus = get_corpus()
    
    # Start with estimated number of words (words are roughly 0.85x tokens)
    estimated_words = int(target_tokens * WORDS_TO_TOKENS_RATIO)
    
    # Generate initial random words
    words = [random.choice(corpus) for _ in range(estimated_words)]
    prompt = " ".join(words)
    
    # Count actual tokens
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    
    # Calculate acceptable bounds
    lower_bound = int(target_tokens * (1 - TOKEN_ERROR_TOLERANCE))
    upper_bound = int(target_tokens * (1 + TOKEN_ERROR_TOLERANCE))
    
    iterations = 0
    while iterations < max_iterations:
        if lower_bound <= actual_tokens <= upper_bound:
            # Within tolerance
            break
            
        if actual_tokens < lower_bound:
            # Need more tokens - add words
            tokens_needed = target_tokens - actual_tokens
            # Estimate words needed (roughly 1 word = 1.2 tokens on average)
            words_to_add = max(1, int(tokens_needed * WORDS_TO_TOKENS_RATIO))
            new_words = [random.choice(corpus) for _ in range(words_to_add)]
            words.extend(new_words)
        else:
            # Too many tokens - remove words
            tokens_excess = actual_tokens - target_tokens
            # Estimate words to remove
            words_to_remove = max(1, int(tokens_excess * WORDS_TO_TOKENS_RATIO))
            words_to_remove = min(words_to_remove, len(words) - 1)  # Keep at least 1 word
            words = words[:-words_to_remove]
        
        # Rebuild prompt and recount
        prompt = " ".join(words)
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        iterations += 1
    
    return prompt, actual_tokens


def parse_token_count(token_str: str) -> int:
    """
    Parse a token count string with optional suffix.
    
    Supports:
    - Plain numbers: "20" -> 20
    - Thousands: "20k" or "20K" -> 20,000
    - Millions: "1m" or "1M" -> 1,000,000
    
    Args:
        token_str: String representing token count
        
    Returns:
        Integer token count
        
    Raises:
        ValueError: If format is invalid
    """
    token_str = token_str.strip().lower()
    
    if not token_str:
        raise ValueError("Empty token count")
    
    # Check for suffix
    multiplier = 1
    if token_str.endswith('k'):
        multiplier = 1_000
        token_str = token_str[:-1]
    elif token_str.endswith('m'):
        multiplier = 1_000_000
        token_str = token_str[:-1]
    
    # Validate that the remaining string is a valid number (digits, optional decimal)
    if not token_str:
        raise ValueError("Missing number before suffix")
    
    # Check for valid number format: digits with optional single decimal point
    has_decimal = False
    for char in token_str:
        if char == '.':
            if has_decimal:
                raise ValueError(f"Invalid token count format: multiple decimal points")
            has_decimal = True
        elif not char.isdigit():
            raise ValueError(f"Invalid token count format: unexpected character '{char}'")
    
    try:
        base_value = float(token_str)
        result = int(base_value * multiplier)
        if result <= 0:
            raise ValueError("Token count must be positive")
        return result
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid token count format: {token_str}") from e


def prompt_for_token_count() -> int:
    """
    Interactively prompt the user for a token count.
    
    Returns:
        Integer token count
    """
    print("\n🎲 Random Token Mode")
    print("-" * 60)
    print("Enter the number of tokens for the random prompt.")
    print("Supported formats:")
    print("  - Plain number: 20 (20 tokens)")
    print("  - Thousands: 20k (20,000 tokens)")
    print("  - Millions: 1m (1,000,000 tokens)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nEnter token count: ").strip()
            token_count = parse_token_count(user_input)
            print(f"✅ Parsed token count: {token_count:,} tokens")
            return token_count
        except ValueError as e:
            print(f"❌ {e}. Please try again.")
        except (EOFError, KeyboardInterrupt):
            print("\n\n❌ Input cancelled")
            raise

