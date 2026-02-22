import requests
import os
from tqdm import tqdm

BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0/"
SAVE_DIR = "data/raw/nvd_bulk"

os.makedirs(SAVE_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

for year in range(2002, 2026):
    filename = f"nvdcve-2.0-{year}.json.gz"
    url = BASE_URL + filename
    save_path = os.path.join(SAVE_DIR, filename)

    print(f"Downloading {filename}...")

    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        print(f"Saved {filename}")
    else:
        print(f"Failed: {filename} | Status: {response.status_code}")
