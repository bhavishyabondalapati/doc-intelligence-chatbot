"""
Search demo (no LLM): type a question and see which chunks the retriever
ranks highest, with scores. Useful for checking retrieval on its own -
if the right chunk isn't near the top here, no prompt will fix the answer.

Run:
    python -m docintel.search [--config hybrid-bge-small+rerank] [--top-k 5]
"""

import argparse
from pathlib import Path

from docintel.config import INDEX_DIR
from docintel.retrieval import DEFAULT_CONFIG, RETRIEVER_CONFIGS, DenseRetriever, make_retriever


def load_retriever(index_dir: Path, config=DEFAULT_CONFIG):
    """Load the saved index and wrap it in the requested retrieval config."""
    if not (index_dir / "manifest.json").exists():
        raise SystemExit(f"No index found in {index_dir}. Run: python -m docintel.pipeline")
    dense, chunks = DenseRetriever.load(index_dir)
    retriever = make_retriever(config, dense=dense)
    retriever.index(chunks)  # dense part is already loaded; this builds BM25 if needed
    return retriever, chunks


def format_hit(chunk, score):
    return f"{chunk['source_file']} p.{chunk['page_start']}-{chunk['page_end']} (score={score:.3f})"


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only search demo.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(RETRIEVER_CONFIGS))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever, chunks = load_retriever(args.index_dir, args.config)
    print(f"Loaded {len(chunks)} chunks ({args.config}). Type a question (or 'quit').\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit", ""):
            break
        result = retriever.search(query, args.top_k)
        print(f"\nconfidence={result.confidence:.3f}")
        for rank, (idx, score) in enumerate(result.hits, start=1):
            chunk = chunks[idx]
            print(f"\n  #{rank} {format_hit(chunk, score)}")
            print(f'  "{chunk["text"][:200]}..."')
        print()


if __name__ == "__main__":
    main()
