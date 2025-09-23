import sys
from typing import Any
from transformers import AutoTokenizer

def count_tokens(model_name: str, context: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(context, add_special_tokens=False))

class SafeList(list):
  def at(self, i: int, default: Any) -> Any:
    if i < 0 or i >= len(self):
      return default
    return self[i]

def main(): 
  args = SafeList(sys.argv)
  model_name = args.at(1, "Qwen/Qwen2-7B-Instruct-AWQ")
  filename = args.at(2, "1984.txt")
  context = open(filename, "r").read()
  tokens_generated = count_tokens(model_name, context)
  print(tokens_generated)

if __name__ == "__main__":
  main()