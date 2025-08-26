# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Single GPU benchmarks:"
	@echo "  kimi              - Run Kimi-K2-Instruct benchmark (tensor parallel size 8)"
	@echo "  qwen235           - Run Qwen3-235B benchmark (tensor parallel size 8)"
	@echo "  qwen30-single     - Run Qwen3-30B benchmark (tensor parallel size 1)"
	@echo "  qwen30-fp8-single - Run Qwen3-30B-FP8 benchmark (tensor parallel size 1)"
	@echo ""
	@echo "Multi-GPU benchmarks:"
	@echo "  qwen30-all-gpus      - Run Qwen3-30B on all GPUs (interactive)"
	@echo "  qwen30-all-gpus-auto - Run Qwen3-30B on all GPUs (automated)"
	@echo "  qwen30-fp8-all-gpus      - Run Qwen3-30B-FP8 on all GPUs (interactive)"
	@echo "  qwen30-fp8-all-gpus-auto - Run Qwen3-30B-FP8 on all GPUs (automated)"
	@echo ""
	@echo "Parallel benchmarks (runs on all GPUs simultaneously):"
	@echo "  qwen30-parallel        - Run Qwen3-30B on all GPUs in parallel"
	@echo "  qwen30-fp8-parallel    - Run Qwen3-30B-FP8 on all GPUs in parallel"
	@echo ""
	@echo "Results analysis:"
	@echo "  list-timestamps       - List available parallel benchmark timestamps"
	@echo "  aggregate-results     - Aggregate results for a specific timestamp"
	@echo ""
	@echo "Utility targets:"
	@echo "  detect-gpus       - Detect available GPUs"
	@echo "  gpu-info          - Show detailed GPU information"
	@echo "  help              - Show this help message"

kimi:
	python main.py \
		--model moonshotai/Kimi-K2-Instruct \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size 8 \
		--max-new-tokens 500 \
		--trust-remote-code

qwen235:
	python main.py \
		--model Qwen/Qwen3-235B-A22B \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size 8 \
		--max-new-tokens 500 \
		--trust-remote-code

qwen30-single:
	python main.py \
		--model Qwen/Qwen3-30B-A3B \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size 1 \
		--max-new-tokens 500 \
		--trust-remote-code

qwen30-fp8-single:
	python main.py \
		--model Qwen/Qwen3-30B-A3B-FP8 \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size 1 \
		--max-new-tokens 500 \
		--trust-remote-code

# Run Qwen 3.0 on all available GPUs with tensor parallel size of 1
qwen30-all-gpus:
	python run_on_all_gpus.py

# Run Qwen 3.0 on all available GPUs (non-interactive, no confirmation prompt)
qwen30-all-gpus-auto:
	python run_on_all_gpus_auto.py

# Run Qwen 3.0-FP8 on all available GPUs with tensor parallel size of 1
qwen30-fp8-all-gpus:
	python run_on_all_gpus.py --model Qwen/Qwen3-30B-A3B-FP8

# Run Qwen 3.0-FP8 on all available GPUs (non-interactive, no confirmation prompt)
qwen30-fp8-all-gpus-auto:
	python run_on_all_gpus_auto.py --model Qwen/Qwen3-30B-A3B-FP8

# Run Qwen 3.0 on all available GPUs in parallel (simultaneous execution)
qwen30-parallel:
	python run_parallel_gpu_benchmarks.py --model Qwen/Qwen3-30B-A3B

# Run Qwen 3.0-FP8 on all available GPUs in parallel (simultaneous execution)
qwen30-fp8-parallel:
	python run_parallel_gpu_benchmarks.py --model Qwen/Qwen3-30B-A3B-FP8

# List available timestamps from parallel benchmark runs
list-timestamps:
	python list_available_timestamps.py

# Aggregate results for a specific timestamp
aggregate-results:
	@if [ -z "$(TIMESTAMP)" ]; then \
		echo "❌ Error: TIMESTAMP parameter is required"; \
		echo "Usage: make aggregate-results TIMESTAMP=20241201_143022"; \
		echo "Use 'make list-timestamps' to see available timestamps"; \
		exit 1; \
	fi
	python aggregate_parallel_results.py $(TIMESTAMP)

# Detect available GPUs
detect-gpus:
	python detect_gpus.py

# Show GPU information
gpu-info:
	@echo "Available GPUs:"
	@python detect_gpus.py
	@echo ""
	@echo "GPU Details:"
	@nvidia-smi --list-gpus 2>/dev/null || echo "nvidia-smi not available"
