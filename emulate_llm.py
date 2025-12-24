#!/usr/bin/env python3
"""
Script to emulate LLM output at a specified tokens per second rate.

This can be used for testing benchmarking infrastructure without requiring
a real LLM model. It streams tokens at a consistent rate to simulate
real LLM behavior.
"""

import asyncio
import argparse
import sys
import time
from typing import AsyncIterator, Optional
import os


class MockLLMStreamer:
    """Emulates LLM output streaming at a specified tokens/second rate"""
    
    def __init__(self, tokens_per_second: float = 25.0, token_delay: Optional[float] = None, source_file: Optional[str] = None):
        """
        Initialize the mock LLM streamer.
        
        Args:
            tokens_per_second: Target tokens per second rate
            token_delay: Optional fixed delay per token (overrides tokens_per_second)
            source_file: Optional path to text file to read tokens from
        """
        if token_delay is not None:
            self.token_delay = token_delay
            self.tokens_per_second = 1.0 / token_delay
        else:
            self.tokens_per_second = tokens_per_second
            self.token_delay = 1.0 / tokens_per_second
        
        # Load source text if provided
        self.source_tokens = None
        if source_file and os.path.exists(source_file):
            with open(source_file, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split into words (simple tokenization)
                self.source_tokens = text.split()
    
    async def generate(self, prompt: str, num_tokens: int = 100) -> AsyncIterator[str]:
        """
        Generate tokens at the specified rate.
        
        Args:
            prompt: Input prompt (not used, but kept for API compatibility)
            num_tokens: Number of tokens to generate
            
        Yields:
            Token strings one at a time
        """
        # Use source tokens if available, otherwise generate mock tokens
        if self.source_tokens:
            tokens_to_yield = self.source_tokens[:num_tokens]
            for token in tokens_to_yield:
                yield token + " "
                await asyncio.sleep(self.token_delay)
        else:
            # Generate mock tokens
            for i in range(num_tokens):
                token = f"token{i} "
                yield token
                await asyncio.sleep(self.token_delay)
    
    async def generate_text(self, prompt: str, num_tokens: int = 100) -> str:
        """
        Generate complete text at the specified rate.
        
        Args:
            prompt: Input prompt
            num_tokens: Number of tokens to generate
            
        Returns:
            Complete generated text
        """
        tokens = []
        async for token in self.generate(prompt, num_tokens):
            tokens.append(token)
        return "".join(tokens)


async def stream_demo(tokens_per_second: float, num_tokens: int, prompt: str, source_file: Optional[str] = None):
    """Demonstrate streaming output at the specified rate"""
    print(f"🚀 Starting mock LLM stream at {tokens_per_second} tokens/second")
    if source_file:
        print(f"📖 Reading from: {source_file}")
    print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")
    print(f"🎯 Generating {num_tokens} tokens\n")
    print("=" * 60)
    
    streamer = MockLLMStreamer(tokens_per_second=tokens_per_second, source_file=source_file)
    
    start_time = time.perf_counter()
    token_count = 0
    
    print("Generated text: ", end="", flush=True)
    
    async for token in streamer.generate(prompt, num_tokens):
        print(token, end="", flush=True)
        token_count += 1
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    actual_tps = token_count / elapsed if elapsed > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"✅ Generation complete!")
    print(f"⏱️  Elapsed time: {elapsed:.2f} seconds")
    print(f"🔢 Tokens generated: {token_count}")
    print(f"⚡ Actual tokens/second: {actual_tps:.2f}")
    print(f"🎯 Target tokens/second: {tokens_per_second:.2f}")
    print(f"📊 Error: {abs(actual_tps - tokens_per_second):.2f} t/s")


async def measure_rate(tokens_per_second: float, duration: float, source_file: Optional[str] = None):
    """Measure the actual rate over a duration"""
    print(f"🔬 Measuring rate: target {tokens_per_second} t/s over {duration}s")
    if source_file:
        print(f"📖 Reading from: {source_file}")
    
    streamer = MockLLMStreamer(tokens_per_second=tokens_per_second, source_file=source_file)
    num_tokens = int(tokens_per_second * duration)
    
    start_time = time.perf_counter()
    token_count = 0
    
    async for _ in streamer.generate("test", num_tokens):
        token_count += 1
    
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    actual_tps = token_count / elapsed if elapsed > 0 else 0
    
    print(f"✅ Measured {token_count} tokens in {elapsed:.2f}s")
    print(f"⚡ Actual rate: {actual_tps:.2f} t/s")
    print(f"📊 Error: {abs(actual_tps - tokens_per_second):.2f} t/s ({abs(actual_tps - tokens_per_second) / tokens_per_second * 100:.1f}%)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Emulate LLM output at a specified tokens per second rate"
    )
    parser.add_argument(
        "--tps",
        type=float,
        default=25.0,
        help="Target tokens per second (default: 25.0)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog.",
        help="Input prompt (default: 'The quick brown fox jumps over the lazy dog.')",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run a rate measurement test instead of streaming demo",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration for rate measurement in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--src",
        type=str,
        default="1984.txt",
        help="Path to text file to read tokens from (default: 1984.txt)",
    )
    
    args = parser.parse_args()
    
    if args.measure:
        asyncio.run(measure_rate(args.tps, args.duration, args.src))
    else:
        asyncio.run(stream_demo(args.tps, args.num, args.prompt, args.src))


if __name__ == "__main__":
    main()

