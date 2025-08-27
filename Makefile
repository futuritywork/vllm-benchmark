# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Model downloads:"
	@echo "  download-qwen - Download Qwen3-30B-A3B and Qwen3-30B-A3B-FP8 to cache"
	@echo "  download      - Download a specific model to cache"
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

# Define function: isnum
# Usage: $(call isnum,<value>)
# Returns: <value> if numeric, -1 otherwise
define isnum
$(if $(filter-out 0 1 2 3 4 5 6 7 8 9,$(firstword $(subst , ,$1))),-1,$1)
endef

setup:
	./setup.sh

main:
	python main.py \
		--model $(MODEL) \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size $(TPAR) \
		--max-new-tokens 500 \
		--trust-remote-code

# Define function: set_tpar
# Usage: $(call set_tpar,<value>)
# Returns: 8 if value is empty, otherwise validates and returns value
define set_tpar
$(if $1,$(call isnum,$1),8)
endef

kimi:
	$(eval TPAR := $(call set_tpar,$(P)))
	make main MODEL=moonshotai/Kimi-K2-Instruct TPAR=$(TPAR)

qwen235:
	$(eval TPAR := $(call set_tpar,$(P)))
	make main MODEL=Qwen/Qwen3-235B-A22B TPAR=$(TPAR)

qwen30:
	$(eval TPAR := $(call set_tpar,$(P)))
	make main MODEL=Qwen/Qwen3-30B-A3B TPAR=$(TPAR)

qwen30-single:
	make qwen30 P=1

qwen30-fp8:
	$(eval TPAR := $(call set_tpar,$(P)))
	make main MODEL=Qwen/Qwen3-30B-A3B-FP8 TPAR=$(TPAR)

qwen30-fp8-single:
	make qwen30-fp8 P=1

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

# Download a specific model to disk
download:
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Error: MODEL parameter is required"; \
		echo "Usage: make download MODEL=Qwen/Qwen3-30B-A3B"; \
		echo "Example: make download MODEL=Qwen/Qwen3-30B-A3B-FP8"; \
		exit 1; \
	fi
	@echo "🚀 Downloading model: $(MODEL)"
	python download_models.py $(MODEL)

# Download Qwen models to disk
download-qwen:
	@echo "🚀 Downloading Qwen models to disk..."
	make download MODEL=Qwen/Qwen3-30B-A3B
	make download MODEL=Qwen/Qwen3-30B-A3B-FP8