"""
Slow: downloads embedding models and runs the real eval. Run with `pytest -m slow`.
Floors sit a few points below the measured numbers (see README) so CPU-vs-GPU
numeric noise doesn't fail CI, but a real regression does.
"""

import pytest

import run_eval

FLOORS = {
    "baseline-minilm": {"hit@4": 0.75, "mrr@10": 0.52},
    "hybrid-bge-small": {"hit@4": 0.85, "mrr@10": 0.75},
}


@pytest.mark.slow
@pytest.mark.parametrize("config", sorted(FLOORS))
def test_retrieval_does_not_regress(eval_corpus, config):
    chunks, questions = eval_corpus
    relevant = run_eval.label_relevance(questions, chunks)
    metrics = run_eval.evaluate_config(config, chunks, questions, relevant)["metrics"]
    for name, floor in FLOORS[config].items():
        assert metrics[name] >= floor, f"{config} {name}={metrics[name]:.2f} < {floor}"


@pytest.mark.slow
def test_hybrid_beats_baseline(eval_corpus):
    chunks, questions = eval_corpus
    relevant = run_eval.label_relevance(questions, chunks)
    base = run_eval.evaluate_config("baseline-minilm", chunks, questions, relevant)["metrics"]
    hybrid = run_eval.evaluate_config("hybrid-bge-small", chunks, questions, relevant)["metrics"]
    assert hybrid["mrr@10"] > base["mrr@10"]
