# vLLM Engine-Direct Connection Ceiling Benchmark

A benchmark to find the highest in-flight concurrency that a vLLM engine can sustain for long prompts while meeting SLA requirements.

## Project Structure

The project has been refactored into multiple modules for better organization:

- **`main.py`** - Main entry point that orchestrates the benchmark
- **`config.py`** - Configuration management and command-line argument parsing
- **`prompt_builder.py`** - Utilities for constructing prompts of target token length
- **`engine_manager.py`** - Engine initialization and sampling parameter creation
- **`benchmark.py`** - Core benchmarking functions (streaming, concurrency testing, ceiling finding)

## Usage

```bash
python3 main.py \
  --model meta-llama/Llama-3.1-8B \
  --target-input-tokens 5000 \
  --ttft-timeout 60 \
  --hold-seconds 5 \
  --sla-ok-rate 0.99 \
  --gpu-device cuda:0
```

## Key Features

- **Direct Engine Connection**: Avoids HTTP/gRPC overhead by driving vLLM's AsyncLLMEngine directly
- **Concurrency Ceiling Detection**: Uses exponential ramp + binary search to find maximum sustainable concurrency
- **SLA Compliance**: Ensures success rate and time-to-first-token (TTFT) thresholds are met
- **Flexible Prompt Building**: Supports both tokenizer-based and character-based prompt construction
- **GPU Selection**: Choose which GPU device to use for inference

## Common Engine Arguments

- `--dtype auto|float16|bfloat16`
- `--tensor-parallel-size 1`
- `--gpu-memory-utilization 0.9`
- `--max-model-len 8192`
- `--max-num-seqs 2048`
- `--gpu-device cuda:0|cuda:1|0|1`

## Output

The benchmark outputs:

- Real-time progress during ramp and search phases
- Final result showing maximum sustainable concurrency
- Detailed results saved to JSON file (default: `engine_conn_results.json`)

## Dependencies

- vLLM
- transformers (optional, for accurate token counting)
- asyncio (built-in)
- argparse (built-in)

Copyright (C) 2025 Alexander Ng
