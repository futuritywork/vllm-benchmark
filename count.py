from transformers import AutoTokenizer

f = open("1984.txt", "r")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-AWQ")

tokens_generated = len(tokenizer.encode(f.read(), add_special_tokens=False))

print(tokens_generated)
