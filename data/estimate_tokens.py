from tokenizers import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer/cyber_tokenizer.json")

file_path = "data/processed/shuffled_corpus.txt"

total_tokens = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        total_tokens += len(tokenizer.encode(line).ids)

print(f"Estimated total tokens: {total_tokens}")
