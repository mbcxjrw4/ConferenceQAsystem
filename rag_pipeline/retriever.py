# retriever.py
# This script will:
# 1. Load the precomputed embeddings (embeddings.npy)
# 2. Load matching metadata (metadata.jsonl)
# 3. Use a similarity search engine (e.g., FAISS)
# 4. Accept a query string
# 5. Return top-k most relevant chunks (and optionally their metadata)

import numpy as np
import faiss
import json
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict

from app.config import (
    EMBEDDINGS_FILE,
    METADATA_FILE,
    BIOBERT_MODEL_NAME,
    DEVICE,
    TOP_K_RETRIEVAL
)

class Retriever:
    def __init__(self, top_k: int = TOP_K_RETRIEVAL):
        self.top_k = top_k
        self.index = None
        self.metadata = []

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME).to(DEVICE)
        self.model.eval()

    def load_index(self):
        print("🔄 Loading embeddings and metadata...")
        embeddings = np.load(EMBEDDINGS_FILE)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f]

    def embed_query(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def retrieve(self, query: str) -> List[Dict]:
        query_vector = self.embed_query(query)
        D, I = self.index.search(query_vector, self.top_k)
        results = [self.metadata[i] for i in I[0]]
        return results
