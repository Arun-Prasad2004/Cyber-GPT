import requests
import json
import os
from tqdm import tqdm
import time

BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
SAVE_PATH = "data/raw/cve_data_full.json"

results_per_page = 2000
start_index = 0
all_vulnerabilities = []

while True:
    print(f"Fetching records starting at {start_index}...")

    params = {
        "resultsPerPage": results_per_page,
        "startIndex": start_index
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code)
        break

    data = response.json()
    vulnerabilities = data.get("vulnerabilities", [])

    if not vulnerabilities:
        break

    all_vulnerabilities.extend(vulnerabilities)

    start_index += results_per_page
    time.sleep(1)  # avoid rate limits

    if start_index >= data.get("totalResults", 0):
        break

print("Total vulnerabilities fetched:", len(all_vulnerabilities))

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump({"vulnerabilities": all_vulnerabilities}, f)

print("Full CVE dataset saved.")
