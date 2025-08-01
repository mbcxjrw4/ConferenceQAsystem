# 🧬 ConferenceQAsystem: Question Answering System for Conference Abstracts

A lightweight retrieval-augmented generation (RAG) pipeline for exploring and querying conference abstracts using local language models. This project is designed for internal use in academic or industrial settings where domain-specific data (e.g., biology, medicine) needs to be explored intelligently.

---

## 🚀 Features

- 🔍 Semantic search over structured abstract data (titles, sessions, content)
- 🧠 Embedding using domain-specific models (e.g. BioBERT)
- 💬 Answer generation using local models like Mistral-7B (GGUF)
- 🖥️ Web UI built with Streamlit for interactive exploration
- 🔧 Modular RAG components (loader, embedder, retriever, generator)

---

## 🗂️ Project Structure
ConferenceQAsystem/ \
├── data/ \
│ ├── raw/ # Original Excel or PDF data \
│ ├── processed/ # Cleaned text chunks (JSONL) \
│ └── embeddings/ # Cached vector embeddings \
├── models/ # Downloaded LLMs and embedding models \
├── rag_pipeline/ # Core RAG pipeline components \
├── scripts/ # Utilities for preprocessing and embedding \
├── app/ # Streamlit app and config \
├── README.md \
└── requirements.txt
