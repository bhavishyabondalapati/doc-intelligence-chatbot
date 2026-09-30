"""
Step 2c: Search Demo (sanity check before building the full chatbot)
------------------------------------------------------------------------
Type a question and see which chunks FAISS thinks are most relevant -
no LLM involved yet, just the raw retrieval step. This is exactly what
happens under the hood right before an LLM would generate an answer in
the full RAG pipeline (that's Step 3).

If the right chunk shows up near the top for something you already know
the answer to, your embeddings + index are working correctly.

Run:
    python src/embed/search_demo.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5   # widened to check whether misses are a ranking problem
            # (right chunk exists, just lower) or a coverage problem
            # (right info isn't in any chunk at all)


def main():
    index_path = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.json"

    if not index_path.exists():
        print("No index found. Run src/embed/build_index.py first.")
        return

    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nLoaded index with {index.ntotal} chunks. Type a question (or 'quit').\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit", ""):
            break

        query_vector = model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype="float32")

        scores, indices = index.search(query_vector, TOP_K)

        print(f"\nTop {TOP_K} matches:")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk = metadata[idx]
            preview = chunk["text"][:200].replace("\n", " ")
            print(f"\n  #{rank} (score={score:.3f}) - {chunk['source_file']} "
                  f"p.{chunk['page_start']}-{chunk['page_end']}")
            print(f'  "{preview}..."')
        print()


if __name__ == "__main__":
    main()