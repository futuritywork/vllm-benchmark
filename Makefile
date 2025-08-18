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
