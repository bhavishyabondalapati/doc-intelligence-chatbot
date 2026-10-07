"""
Evaluation helpers: how good is retrieval, and when should we abstain?

Relevance labels. Each answerable question in eval/questions.json lists
its source file and one or more short "evidence" quotes copied from the
PDF. A chunk counts as relevant if it comes from that file and contains
one of the quotes. Matching is fuzzy (rapidfuzz partial_ratio >= 90 on
lowercased, whitespace-normalised text) so small OCR errors on scanned
pages don't break it. Because labels are quotes - not chunk ids - they
stay valid if we change the chunk size or the parser.

Retrieval metrics (answerable questions only):
  Hit@k  - fraction of questions with at least one relevant chunk in the
           top k results. With k = 4 (what the chatbot sends to the
           LLM) this is "how often can the LLM possibly answer?"
  MRR@k  - mean of 1/rank of the first relevant chunk (0 if none in the
           top k). Rewards putting the right chunk first, not just
           somewhere in the list.

Abstention (all questions): the retriever's confidence score decides
whether to answer. Treating "abstain" as the positive class, we sweep
every possible threshold, compute precision/recall, and pick the one
with the best F1.
"""

import re

import numpy as np
from rapidfuzz import fuzz

MATCH_THRESHOLD = 90


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def chunk_matches(chunk_text, evidence_quotes, threshold=MATCH_THRESHOLD):
    text = normalize(chunk_text)
    return any(fuzz.partial_ratio(normalize(q), text) >= threshold for q in evidence_quotes)


def relevant_chunk_ids(question, chunks):
    """Indices of chunks that are relevant to an answerable question."""
    return {
        i for i, chunk in enumerate(chunks)
        if chunk["source_file"] == question["source_file"]
        and chunk_matches(chunk["text"], question["evidence"])
    }


def first_relevant_rank(ranked_ids, relevant_ids):
    """1-based rank of the first relevant chunk, or None if absent."""
    for rank, idx in enumerate(ranked_ids, start=1):
        if idx in relevant_ids:
            return rank
    return None


def retrieval_metrics(ranks, ks=(1, 3, 5, 10), mrr_k=10):
    """ranks: first-relevant rank (or None) per answerable question."""
    n = len(ranks)
    metrics = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in ks}
    metrics[f"mrr@{mrr_k}"] = sum(1 / r for r in ranks if r is not None and r <= mrr_k) / n
    return metrics


def precision_recall_curve(confidences, should_abstain):
    """
    For each candidate threshold t, we abstain when confidence < t.
    Returns a list of dicts: threshold, precision, recall, f1 (for the
    "abstain" class), plus answered_rate on answerable questions.
    """
    conf = np.asarray(confidences, dtype=float)
    positive = np.asarray(should_abstain, dtype=bool)
    # Candidate thresholds: just above each observed score, plus "never abstain".
    candidates = np.unique(np.concatenate([[conf.min()], np.nextafter(conf, np.inf)]))
    curve = []
    for t in candidates:
        abstain = conf < t
        tp = int(np.sum(abstain & positive))
        fp = int(np.sum(abstain & ~positive))
        fn = int(np.sum(~abstain & positive))
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        answered = float(np.mean(~abstain[~positive])) if (~positive).any() else 0.0
        curve.append({"threshold": float(t), "precision": precision, "recall": recall,
                      "f1": f1, "answered_rate": answered})
    return curve


def pick_threshold(confidences, should_abstain):
    """Threshold with the best abstention F1 (ties -> lower threshold, i.e. answer more)."""
    curve = precision_recall_curve(confidences, should_abstain)
    return max(curve, key=lambda point: (point["f1"], -point["threshold"]))


def roc_auc(confidences, should_abstain):
    """
    Probability that a random answerable question gets a higher confidence
    than a random unanswerable one (1.0 = perfectly separable, 0.5 = coin
    flip). Threshold-free, so it compares retrievers fairly.
    """
    conf = np.asarray(confidences, dtype=float)
    positive = np.asarray(should_abstain, dtype=bool)
    answerable, unanswerable = conf[~positive], conf[positive]
    if not len(answerable) or not len(unanswerable):
        return float("nan")
    wins = (answerable[:, None] > unanswerable[None, :]).sum()
    ties = (answerable[:, None] == unanswerable[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(answerable) * len(unanswerable)))


def cross_validated_f1(confidences, should_abstain, folds=5, seed=0):
    """
    Honest estimate of abstention F1: pick the threshold on 4/5 of the
    questions, score it on the held-out 1/5, repeat, pool the predictions.
    (Picking and scoring on the same 60 questions would be optimistic.)
    """
    conf = np.asarray(confidences, dtype=float)
    positive = np.asarray(should_abstain, dtype=bool)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(conf), dtype=int)
    for cls in (True, False):  # stratify so every fold gets some unanswerables
        idx = rng.permutation(np.flatnonzero(positive == cls))
        fold_of[idx] = np.arange(len(idx)) % folds

    predicted = np.zeros(len(conf), dtype=bool)
    for f in range(folds):
        train, test = fold_of != f, fold_of == f
        t = pick_threshold(conf[train], positive[train])["threshold"]
        predicted[test] = conf[test] < t

    tp = int(np.sum(predicted & positive))
    precision = tp / predicted.sum() if predicted.sum() else 1.0
    recall = tp / positive.sum() if positive.sum() else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """
    95% confidence interval for a mean by bootstrapping: resample the
    questions with replacement many times and look at the spread of the
    mean. With only 48 answerable questions, one question moves Hit@1 by
    ~2 points, so differences smaller than the interval are noise.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n_boot=10000, alpha=0.05, seed=0):
    """
    Mean of (a - b) over the same questions, with a 95% bootstrap interval.
    Pairing matters: both systems answer the same questions, so we resample
    questions (not each system separately). If the interval excludes 0,
    the difference is unlikely to be luck.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    low, high = bootstrap_ci(diff, n_boot, alpha, seed)
    return float(diff.mean()), low, high
