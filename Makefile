kimi:
	python main.py \
		--model moonshotai/Kimi-K2-Instruct \
		--max-concurrency-cap 1024 \
		--start-concurrency 2 \
		--log-output \
		--tensor-parallel-size 8 \
		--max-num-seqs 1024 \
		--max-new-tokens 500 \
		--trust-remote-code
