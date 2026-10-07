import pytest

from docintel.evaluation import (
    chunk_matches, cross_validated_f1, first_relevant_rank, pick_threshold, precision_recall_curve,
    relevant_chunk_ids, retrieval_metrics, roc_auc,
)


def test_fuzzy_evidence_match_tolerates_ocr_noise():
    chunk = "The recommended starting dose is 50 mg once dai1y, with or without food."
    assert chunk_matches(chunk, ["starting dose is 50 mg once daily"])
    assert not chunk_matches(chunk, ["maximum dose is 200 mg per day"])


def test_relevance_requires_same_source_file():
    chunks = [{"source_file": "a.pdf", "text": "take with food"},
              {"source_file": "b.pdf", "text": "take with food"}]
    q = {"source_file": "b.pdf", "evidence": ["take with food"]}
    assert relevant_chunk_ids(q, chunks) == {1}


def test_ranks_and_metrics():
    assert first_relevant_rank([5, 2, 9], {9, 2}) == 2
    assert first_relevant_rank([5], {1}) is None
    m = retrieval_metrics([1, 3, None, 12], ks=(1, 5), mrr_k=10)
    assert m["hit@1"] == 0.25 and m["hit@5"] == 0.5
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3) / 4)


def test_threshold_separates_perfectly_separable_scores():
    conf = [0.9, 0.8, 0.7, 0.2, 0.1]
    abstain = [False, False, False, True, True]
    best = pick_threshold(conf, abstain)
    assert best["f1"] == 1.0 and 0.2 < best["threshold"] <= 0.7
    assert roc_auc(conf, abstain) == 1.0


def test_curve_includes_never_abstain_point():
    curve = precision_recall_curve([0.5, 0.4], [False, True])
    never = min(curve, key=lambda p: p["threshold"])
    assert never["recall"] == 0.0 and never["answered_rate"] == 1.0


def test_cross_validated_f1_is_perfect_when_separable():
    conf = [0.9] * 10 + [0.1] * 5
    abstain = [False] * 10 + [True] * 5
    assert cross_validated_f1(conf, abstain, folds=5) == 1.0


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_n():
    from docintel.evaluation import bootstrap_ci, paired_bootstrap_diff
    small = bootstrap_ci([0, 1] * 10)
    large = bootstrap_ci([0, 1] * 500)
    assert small[0] < 0.5 < small[1]
    assert (large[1] - large[0]) < (small[1] - small[0])
    mean, low, high = paired_bootstrap_diff([1] * 30, [0] * 30)
    assert mean == 1.0 and low == 1.0 and high == 1.0
