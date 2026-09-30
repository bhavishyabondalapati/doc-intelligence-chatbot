"""
Step 2a: Chunking
------------------
Step 1 gave us "blocks" of text (one JSON file per PDF, in reading order
down the page). Blocks are often too small on their own to carry much
meaning (e.g. "Class : Economy"), and a single giant chunk mixes too many
unrelated ideas together, which hurts retrieval accuracy later.

So this step glues blocks back together, in order, into chunks of a
target size (measured in words), with a little overlap between
consecutive chunks. The overlap matters: if a sentence gets cut in half
right at a chunk boundary, overlap means both chunks still contain
enough of it to be useful when searched later.

Input:  data/processed/*.json      (from Step 1)
Output: data/chunks/*.chunks.json  (ready to embed in Step 2b)

Run:
    python src/embed/chunk_documents.py
"""

import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CHUNKS_DIR = Path(__file__).resolve().parents[2] / "data" / "chunks"

TARGET_CHUNK_WORDS = 120   # smaller chunks = each vector represents one
                           # focused idea instead of several blended together
OVERLAP_WORDS = 30         # carry the last 30 words into the next chunk


def chunk_blocks(blocks):
    """
    Walk through blocks in order, accumulating words until we hit the
    target chunk size, then start a new chunk - carrying the tail of the
    previous chunk forward so context isn't lost at the seam.
    """
    chunks = []
    current_words = []
    current_pages = set()

    def flush():
        if not current_words:
            return None
        pages = sorted(current_pages)
        return {
            "text": " ".join(current_words),
            "page_start": pages[0],
            "page_end": pages[-1],
        }

    for block in blocks:
        current_words.extend(block["text"].split())
        current_pages.add(block["page"])

        if len(current_words) >= TARGET_CHUNK_WORDS:
            chunks.append(flush())
            # Carry the tail forward as overlap for the next chunk
            current_words = current_words[-OVERLAP_WORDS:]
            current_pages = {block["page"]}

    # Only keep a trailing leftover if it has real new content beyond
    # just the carried-over overlap (avoids a near-duplicate tiny chunk).
    final = flush()
    if final and (not chunks or len(current_words) > OVERLAP_WORDS):
        chunks.append(final)

    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = sorted(PROCESSED_DIR.glob("*.json"))

    if not json_files:
        print(f"No processed files found in {PROCESSED_DIR}. Run Step 1 first.")
        return

    total_chunks = 0
    for json_path in json_files:
        with open(json_path) as f:
            blocks = json.load(f)

        source_file = blocks[0]["source_file"] if blocks else json_path.stem
        chunks = chunk_blocks(blocks)

        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{json_path.stem}_{i}"
            chunk["source_file"] = source_file

        out_path = CHUNKS_DIR / f"{json_path.stem}.chunks.json"
        with open(out_path, "w") as f:
            json.dump(chunks, f, indent=2)

        print(f"{source_file}: {len(blocks)} blocks -> {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} total chunks saved to {CHUNKS_DIR}")


if __name__ == "__main__":
    main()