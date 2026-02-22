import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import re

INPUT_FOLDERS = [
    "data/security.stackexchange.com",
    "data/serverfault.com"
]

OUTPUT_FILE = "data/processed/stackexchange_corpus.txt"

os.makedirs("data/processed", exist_ok=True)

def clean_html(raw_html):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', raw_html)
    return " ".join(text.split())

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

    for folder in INPUT_FOLDERS:
        posts_path = os.path.join(folder, "Posts.xml")

        print(f"Processing {posts_path}")

        context = ET.iterparse(posts_path, events=("end",))

        for event, elem in tqdm(context):
            if elem.tag == "row":
                body = elem.attrib.get("Body", "")
                title = elem.attrib.get("Title", "")

                if body:
                    cleaned = clean_html(title + " " + body)
                    if len(cleaned) > 100:
                        out_file.write(cleaned + "\n\n")

                elem.clear()

print("StackExchange corpus built.")
