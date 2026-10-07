"""
Run the whole indexing pipeline in one go: parse -> chunk -> embed + index.

Run:
    python -m docintel.pipeline [--pdf-dir data/raw_pdfs] [--config hybrid-bge-small+rerank]
"""

import argparse
from pathlib import Path

from docintel.build_index import build_index
from docintel.chunking import chunk_folder
from docintel.config import CHUNKS_DIR, INDEX_DIR, PROCESSED_DIR, RAW_PDF_DIR
from docintel.ingest import ingest_folder
from docintel.retrieval import DEFAULT_CONFIG, RETRIEVER_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Parse, chunk and index a folder of PDFs.")
    parser.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(RETRIEVER_CONFIGS))
    args = parser.parse_args()

    print("== 1/3 Parsing PDFs")
    if not ingest_folder(args.pdf_dir, PROCESSED_DIR):
        return
    print("\n== 2/3 Chunking")
    chunk_folder(PROCESSED_DIR, CHUNKS_DIR)
    print("\n== 3/3 Embedding + indexing")
    build_index(CHUNKS_DIR, INDEX_DIR, args.config)
    print("\nDone. Chat with: python -m docintel.chat")


if __name__ == "__main__":
    main()
