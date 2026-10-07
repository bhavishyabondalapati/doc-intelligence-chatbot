import shutil

import pytest

from docintel.evaluation import relevant_chunk_ids

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract for the scanned PDF")


def test_question_set_shape(eval_corpus):
    _, questions = eval_corpus
    assert len(questions) == 60
    assert sum(not q["answerable"] for q in questions) == 12
    assert len({q["id"] for q in questions}) == 60


def test_every_answerable_question_has_findable_evidence(eval_corpus):
    """Guards against typos in labels and against parser changes that lose text (incl. OCR)."""
    chunks, questions = eval_corpus
    missing = [q["id"] for q in questions if q["answerable"] and not relevant_chunk_ids(q, chunks)]
    assert missing == []


def test_scanned_pdf_went_through_ocr(eval_corpus):
    chunks, _ = eval_corpus
    scanned = [c for c in chunks if c["source_file"].endswith("_SCANNED.pdf")]
    assert scanned, "scanned PDF produced no chunks"
    assert all(r["bbox"]["x1"] <= 612 for c in scanned for r in c["regions"])  # PDF points, not pixels
