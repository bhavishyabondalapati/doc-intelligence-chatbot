"""
Shared settings: where data lives and which models we use.

Keeping these in one place means the indexer, the chatbot and the
evaluation can't silently disagree (e.g. the index built with one
embedding model but queried with another).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"

# Baseline embedding model (the original project's choice).
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking (in words). See chunking.py for why these numbers.
TARGET_CHUNK_WORDS = 120
OVERLAP_WORDS = 30


def best_device():
    """Use Apple Silicon's GPU ("mps") when available, else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
