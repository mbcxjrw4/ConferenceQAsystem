# generate_embeddings.py
# This script will:
# 1. Load the split text chunks from chunks.jsonl
# 2. Encodes them using a BioBERT model
# 3. Stores the resulting embeddings and metadata for indexing in FAISS or other DBs

import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Configuration
INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_EMBEDDINGS = Path("data/embeddings/embeddings.npy")
OUTPUT_METADATA = Path("data/embeddings/metadata.jsonl")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"  # Or your preferred model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return embedding.cpu().numpy().squeeze()

def main():
    OUTPUT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)

    embeddings = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(OUTPUT_METADATA, "w", encoding="utf-8") as f_meta:
        for line in tqdm(f_in, desc="Embedding chunks"):
            item = json.loads(line)
            vector = get_embedding(item["text"])
            embeddings.append(vector)

            # Save metadata for retrieval reference
            json.dump({
                "chunk_id": item["chunk_id"],
                "title": item["title"],
                "session": item["session"],
                "text": item["text"],
                "meta": item["meta"]
            }, f_meta, ensure_ascii=False)
            f_meta.write("\n")

    # Save as .npy
    np.save(OUTPUT_EMBEDDINGS, np.array(embeddings))
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_EMBEDDINGS}")

if __name__ == "__main__":
    main()

