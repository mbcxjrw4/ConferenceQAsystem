from pathlib import Path

# Paths
DATA_DIR = Path("data/")
RAW_DATA_DIR = DATA_DIR / "raw/"
PROCESSED_DATA_DIR = DATA_DIR / "processed/"
EMBEDDINGS_DIR = DATA_DIR / "embeddings/"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.jsonl"

# Models
BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MISTRAL_MODEL_PATH = Path("models/mistral-7b/mistral-7b.Q4_K_M.gguf")

# Inference
DEVICE = "cuda"  # or "cpu"
TOP_K_RETRIEVAL = 5
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_K_SAMPLING = 40
