# split_chucks.py
# This script will:
# 1. Read the abstracts.jsonl you prepared
# 2. Split the "text" field into overlapping chunks based on token count (suitable for embedding)
# 3. Retain metadata like title, session, and authors for traceability

import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer

# Configurable parameters
INPUT_FILE = Path("data/processed/abstracts.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")
CHUNK_SIZE = 200  # tokens per chunk
CHUNK_OVERLAP = 50
TOKENIZER_NAME = "bert-base-uncased"  # use same tokenizer as embedding model

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def split_text_into_chunks(text: str, max_tokens: int, overlap: int) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            doc = json.loads(line)
            chunks = split_text_into_chunks(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "chunk_id": f'{doc["meta"]["id"]}_chunk{i}',
                    "title": doc["title"],
                    "session": doc["session"],
                    "authors": doc["authors"],
                    "text": chunk,
                    "meta": {
                        "source_id": doc["meta"]["id"],
                        "chunk_index": i
                    }
                }
                json.dump(chunk_doc, f_out, ensure_ascii=False)
                f_out.write("\n")

if __name__ == "__main__":
    main()

