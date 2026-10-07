"""
Step 3: Embeddings + FAISS Index
-----------------------------------
This is where text becomes something a computer can search "by meaning."

An embedding model reads a chunk of text and outputs a vector (384
numbers for the small models here) representing its meaning - chunks
that mean similar things end up as vectors close together in that space.
FAISS (built by Meta) then lets us ask "which chunks are closest to this
question's vector?" across thousands of chunks in milliseconds.

The embedding model comes from the retrieval config (see retrieval.py);
the default config is the one that scored best in eval/. The model name
is saved next to the index, so the chatbot can't query it with a
different model by accident.

Note: the first run downloads model weights from HuggingFace (needs
internet once). After that they're cached locally and work offline.

Input:  data/chunks/*.chunks.json
Output: data/index/{faiss.index, metadata.json, manifest.json}

Run:
    python -m docintel.build_index [--config hybrid-bge-small]
"""

import argparse
from pathlib import Path

from docintel.chunking import load_chunks
from docintel.config import CHUNKS_DIR, INDEX_DIR
from docintel.retrieval import DEFAULT_CONFIG, RETRIEVER_CONFIGS, DenseRetriever, embed_model_for


def build_index(chunks_dir: Path, index_dir: Path, config=DEFAULT_CONFIG):
    chunks = load_chunks(chunks_dir)
    if not chunks:
        print(f"No chunk files found in {chunks_dir}. Run docintel.chunking first.")
        return None
    model_name = embed_model_for(config)
    if model_name is None:
        raise ValueError(f"Config {config!r} has no embedding model; pick a dense or hybrid config.")

    print(f"Embedding {len(chunks)} chunks with {model_name}...")
    dense = DenseRetriever(model_name).index(chunks)
    dense.save(index_dir, chunks)
    print(f"Indexed {dense.faiss_index.ntotal} chunks -> {index_dir}")
    return dense


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and build the FAISS index.")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(RETRIEVER_CONFIGS))
    args = parser.parse_args()
    build_index(args.chunks_dir, args.index_dir, args.config)


if __name__ == "__main__":
    main()
