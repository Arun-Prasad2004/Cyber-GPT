import os
from tqdm import tqdm

INPUT_FILES = [
    "data/processed/full_corpus.txt",
    "data/processed/cwe_corpus.txt",
    "data/processed/stackexchange_corpus.txt"
]

OUTPUT_FILE = "data/processed/final_cyber_corpus.txt"

seen = set()
os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
    for file in INPUT_FILES:
        if os.path.exists(file):
            print(f"Merging {file}")
            with open(file, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    text = line.strip()
                    if len(text) > 50 and text not in seen:
                        out_file.write(text + "\n")
                        seen.add(text)

print("Final corpus built.")

size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"Final corpus size: {size_mb:.2f} MB")
