import gzip
import os
import shutil
from tqdm import tqdm

INPUT_DIR = "data/data/raw/nvd_bulk"
OUTPUT_DIR = "data/data/raw/nvd_extracted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".gz")]

for file in tqdm(files):
    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file.replace(".gz", ""))

    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print("Extraction complete.")
