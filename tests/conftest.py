import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))


@pytest.fixture(scope="session")
def eval_corpus():
    """Parsed + chunked eval corpus and its questions (parsing is cached in eval/.cache)."""
    import json

    import run_eval
    chunks = run_eval.load_corpus_chunks()
    questions = json.loads(run_eval.QUESTIONS_PATH.read_text())
    return chunks, questions
