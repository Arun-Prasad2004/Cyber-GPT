import random

INPUT_FILE = "data/processed/final_cyber_corpus.txt"
OUTPUT_FILE = "data/processed/shuffled_corpus.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Corpus shuffled.")
