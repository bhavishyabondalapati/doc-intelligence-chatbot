"""
Step 2: Chunking
-----------------
Step 1 gave us "blocks" of text (one JSON file per PDF, in reading order
down the page). Blocks are often too small on their own to carry much
meaning (e.g. "Class : Economy"), and a single giant chunk mixes too many
unrelated ideas together, which hurts retrieval later.

So this step glues blocks back together, in order, into chunks of a
target size (measured in words), with a little overlap between
consecutive chunks. The overlap matters: if a sentence gets cut in half
right at a chunk boundary, overlap means both chunks still contain
enough of it to be useful when searched later.

Each chunk also keeps a list of "regions" - the page + bounding box of
every block its words came from - so an answer can point back to the
exact spot on the page, not just the page number.

Input:  data/processed/*.json      (from Step 1)
Output: data/chunks/*.chunks.json  (ready to embed in Step 3)

Run:
    python -m docintel.chunking [--in-dir data/processed] [--out-dir data/chunks]
"""

import argparse
import json
from pathlib import Path

from docintel.config import CHUNKS_DIR, OVERLAP_WORDS, PROCESSED_DIR, TARGET_CHUNK_WORDS


def chunk_blocks(blocks, target_words=TARGET_CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """
    Walk through blocks in order, accumulating words until we hit the
    target chunk size, then start a new chunk - carrying the tail of the
    previous chunk forward so context isn't lost at the seam.

    Every word remembers which block it came from, so a chunk's pages and
    regions are exactly the blocks its words came from (including the
    blocks the carried-over overlap words belong to).
    """
    chunks = []
    words = []  # list of (word, block_index)

    def make_chunk():
        block_ids = sorted({b for _, b in words})
        pages = [blocks[b]["page"] for b in block_ids]
        return {
            "text": " ".join(w for w, _ in words),
            "page_start": min(pages),
            "page_end": max(pages),
            "regions": [{"page": blocks[b]["page"], "bbox": blocks[b]["bbox"]} for b in block_ids],
        }

    new_words = 0  # words added since the last chunk was cut
    for block_index, block in enumerate(blocks):
        block_words = block["text"].split()
        words.extend((w, block_index) for w in block_words)
        new_words += len(block_words)

        if len(words) >= target_words:
            chunks.append(make_chunk())
            words = words[-overlap_words:] if overlap_words else []
            new_words = 0

    # Only keep a trailing leftover if it has real new content beyond the
    # carried-over overlap (avoids a near-duplicate tiny chunk).
    if words and (not chunks or new_words > 0):
        chunks.append(make_chunk())

    return chunks


def chunk_folder(in_dir: Path, out_dir: Path, verbose=True, **chunk_kwargs):
    """Chunk every blocks JSON in in_dir; returns all chunks (with ids and source file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for json_path in sorted(in_dir.glob("*.json")):
        blocks = json.loads(json_path.read_text())
        source_file = blocks[0]["source_file"] if blocks else json_path.stem
        chunks = chunk_blocks(blocks, **chunk_kwargs)
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{json_path.stem}_{i}"
            chunk["source_file"] = source_file

        (out_dir / f"{json_path.stem}.chunks.json").write_text(json.dumps(chunks, indent=2))
        if verbose:
            print(f"{source_file}: {len(blocks)} blocks -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def load_chunks(chunks_dir: Path):
    chunks = []
    for path in sorted(chunks_dir.glob("*.chunks.json")):
        chunks.extend(json.loads(path.read_text()))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk parsed PDF blocks.")
    parser.add_argument("--in-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=CHUNKS_DIR)
    args = parser.parse_args()
    chunks = chunk_folder(args.in_dir, args.out_dir)
    if not chunks:
        print(f"No processed files found in {args.in_dir}. Run docintel.ingest first.")
    else:
        print(f"\nDone. {len(chunks)} total chunks saved to {args.out_dir}")


if __name__ == "__main__":
    main()
