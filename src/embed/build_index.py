"""
Step 2b: Embeddings + FAISS Index
-----------------------------------
This is where text becomes something a computer can search "by meaning."

An embedding model reads a chunk of text and outputs a vector (here, 384
numbers) representing its meaning - chunks that mean similar things end
up as vectors that sit close together in that 384-dimensional space.
We're using `all-MiniLM-L6-v2` from HuggingFace: small (~80MB), fast,
runs great on CPU, and it's the standard starter choice for this kind of
project - not the fanciest model, but good enough to prove retrieval
works, and easy to swap out later.

Once every chunk has a vector, we hand them all to FAISS (built by
Meta) - a library for fast nearest-neighbor search. FAISS lets us ask
"which chunks are closest in meaning to this new vector?" out of
thousands of stored vectors, in milliseconds.

Note: the first run downloads the model weights from HuggingFace (needs
internet once). After that it's cached locally and works offline.

Input:  data/chunks/*.chunks.json   (from Step 2a)
Output: data/index/faiss.index      (the searchable vector index)
        data/index/metadata.json    (maps each vector back to its chunk)

Run:
    python src/embed/build_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "chunks"
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_all_chunks():
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.chunks.json")):
        with open(path) as f:
            chunks.extend(json.load(f))
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_all_chunks()

    if not chunks:
        print(f"No chunk files found in {CHUNKS_DIR}. Run Step 2a first.")
        return

    print(f"Loaded {len(chunks)} chunks. Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks (first run also downloads the model)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    # Inner product on normalized vectors = cosine similarity search
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"\nIndexed {index.ntotal} chunks (dimension={dimension})")
    print(f"Saved index to {INDEX_DIR / 'faiss.index'}")
    print(f"Saved metadata to {INDEX_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
