# generator.py
# This script will:
# 1. take: A user query; The top-k retrieved chunks; A language model (local or remote)
# 2. return A generated answer based on the context

from llama_cpp import Llama
from typing import List, Dict
from app.config import (
    MISTRAL_MODEL_PATH,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K_SAMPLING
)

# Load model once
llm = Llama(
    model_path=str(MISTRAL_MODEL_PATH),
    n_ctx=2048,
    n_threads=8,
    temperature=TEMPERATURE,
    top_k=TOP_K_SAMPLING
)

def format_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join(f"- {chunk['text']}" for chunk in retrieved_chunks)
    prompt = (
        "You are a scientific assistant specialized in biology conference material.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    prompt = format_prompt(query, retrieved_chunks)
    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["\n\n", "</s>"])
    return output["choices"][0]["text"].strip()
