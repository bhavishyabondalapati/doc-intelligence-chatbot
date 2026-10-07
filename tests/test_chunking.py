from docintel.chunking import chunk_blocks


def make_blocks(word_counts, page=1):
    """Blocks whose words are w0, w1, ... so we can check order and overlap."""
    blocks, n = [], 0
    for i, count in enumerate(word_counts):
        words = [f"w{n + j}" for j in range(count)]
        n += count
        blocks.append({"text": " ".join(words), "page": page + i // 2,
                       "bbox": {"x0": i, "y0": i, "x1": i + 1, "y1": i + 1}})
    return blocks


def test_chunks_respect_target_size_and_overlap():
    blocks = make_blocks([10] * 10)  # 100 words
    chunks = chunk_blocks(blocks, target_words=40, overlap_words=10)
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert len(first) == 40
    assert second[:10] == first[-10:]  # overlap carried forward
    # every original word appears in some chunk, in order
    seen = [w for c in chunks for w in c["text"].split()]
    assert set(seen) == {f"w{i}" for i in range(100)}


def test_chunks_keep_pages_and_regions():
    blocks = make_blocks([30, 30, 30])  # pages 1, 1, 2
    chunks = chunk_blocks(blocks, target_words=50, overlap_words=5)
    assert chunks[0]["page_start"] == 1 and chunks[0]["page_end"] == 1
    assert chunks[1]["page_end"] == 2
    # the second chunk includes the overlap words from block 1, so it
    # lists block 1's region as well as block 2's
    assert [r["bbox"]["x0"] for r in chunks[1]["regions"]] == [1, 2]


def test_no_tiny_duplicate_trailing_chunk():
    blocks = make_blocks([50])
    chunks = chunk_blocks(blocks, target_words=50, overlap_words=10)
    assert len(chunks) == 1  # leftover is only the overlap -> dropped


def test_short_document_still_gives_one_chunk():
    chunks = chunk_blocks(make_blocks([5]), target_words=120, overlap_words=30)
    assert len(chunks) == 1 and chunks[0]["text"].split() == [f"w{i}" for i in range(5)]


def test_empty_input():
    assert chunk_blocks([]) == []
