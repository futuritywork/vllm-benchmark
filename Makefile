# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Single GPU benchmarks:"
	@echo "  kimi              - Run Kimi-K2-Instruct benchmark (tensor parallel size 8)"
	@echo "  qwen235           - Run Qwen3-235B benchmark (tensor parallel size 8)"
	@echo "  qwen30            - Run Qwen3-30B benchmark (tensor parallel size 8)"
	@echo "  qwen30-single     - Run Qwen3-30B benchmark (tensor parallel size 1)"
	@echo ""
	@echo "Multi-GPU benchmarks:"
	@echo "  qwen30-all-gpus      - Run Qwen3-30B on all GPUs (interactive)"
	@echo "  qwen30-all-gpus-auto - Run Qwen3-30B on all GPUs (automated)"
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

qwen30:
	python main.py \
		--model Qwen/Qwen3-30B-A3B \
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

# Run Qwen 3.0 on all available GPUs with tensor parallel size of 1
qwen30-all-gpus:
	python run_on_all_gpus.py

# Run Qwen 3.0 on all available GPUs (non-interactive, no confirmation prompt)
qwen30-all-gpus-auto:
	python run_on_all_gpus_auto.py

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
