"""
Retrieval + abstention evaluation on a public medical corpus.

Corpus:    eval/corpus/  - 5 FDA drug labels from DailyMed + 1 simulated
           scanned label (see eval/download_corpus.py, eval/make_scanned_pdf.py)
Questions: eval/questions.json - 60 hand-labelled questions, 48 answerable
           (each with evidence quotes) and 12 unanswerable.

For every retrieval config in docintel.retrieval.RETRIEVER_CONFIGS this:
  1. indexes the corpus chunks,
  2. retrieves the top 10 chunks for each question,
  3. scores Hit@k / MRR on the answerable questions,
  4. sweeps the abstention threshold on the confidence score, picks the
     best-F1 threshold from the precision-recall curve, and also reports a
     5-fold cross-validated F1 (threshold chosen on 4/5, scored on 1/5).

Outputs (eval/results/):
  results.md          - the comparison table (copied into the README)
  results.json        - all metrics, thresholds, PR curves, per-question ranks
  pr_curves.png       - precision-recall curves for abstention
  labels_review.md    - every question + label + where the evidence was found

Run (from the project root):
    python eval/run_eval.py                      # all configs
    python eval/run_eval.py --configs baseline-minilm bm25
"""

import argparse
import json
import time
from pathlib import Path

from docintel.chunking import chunk_folder
from docintel.evaluation import (
    bootstrap_ci, cross_validated_f1, paired_bootstrap_diff, first_relevant_rank, pick_threshold, precision_recall_curve,
    relevant_chunk_ids, retrieval_metrics, roc_auc,
)
from docintel.ingest import ingest_folder
from docintel.retrieval import RETRIEVER_CONFIGS, make_retriever

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
CACHE_DIR = EVAL_DIR / ".cache"
RESULTS_DIR = EVAL_DIR / "results"

TOP_K = 10
CHAT_K = 4  # chunks the chatbot sends to the LLM


def load_corpus_chunks():
    """Parse (cached - OCR is the slow part) and chunk the eval corpus."""
    blocks_dir = CACHE_DIR / "blocks"
    pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    cached = [blocks_dir / f"{p.stem}.json" for p in pdfs]
    if not all(c.exists() and c.stat().st_mtime > p.stat().st_mtime for c, p in zip(cached, pdfs)):
        print("Parsing corpus (first run OCRs the scanned PDF)...")
        ingest_folder(CORPUS_DIR, blocks_dir)
    return chunk_folder(blocks_dir, CACHE_DIR / "chunks", verbose=False)


def label_relevance(questions, chunks):
    """Map question id -> set of relevant chunk indices. Fails loudly on a broken label."""
    relevant = {}
    for q in questions:
        if not q["answerable"]:
            continue
        relevant[q["id"]] = relevant_chunk_ids(q, chunks)
        if not relevant[q["id"]]:
            raise SystemExit(f"Label error: no chunk in {q['source_file']} matches the evidence "
                             f"for {q['id']} ({q['question']!r}). Fix eval/questions.json.")
    return relevant


def evaluate_config(config, chunks, questions, relevant):
    retriever = make_retriever(config)
    start = time.perf_counter()
    retriever.index(chunks)
    index_seconds = time.perf_counter() - start

    rows, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        result = retriever.search(q["question"], TOP_K)
        latencies.append(time.perf_counter() - start)
        ranked = [idx for idx, _ in result.hits]
        rank = first_relevant_rank(ranked, relevant[q["id"]]) if q["answerable"] else None
        rows.append({"id": q["id"], "answerable": q["answerable"], "rank": rank,
                     "confidence": result.confidence, "top_chunk": chunks[ranked[0]]["chunk_id"]})

    answerable = [r for r in rows if r["answerable"]]
    metrics = retrieval_metrics([r["rank"] for r in answerable], ks=(1, 3, 4, 5, 10))
    for r in answerable:
        r["rr"] = 1 / r["rank"] if r["rank"] else 0.0
    metrics["mrr@10_ci"] = bootstrap_ci([r["rr"] for r in answerable])
    metrics["hit@1_ci"] = bootstrap_ci([r["rank"] == 1 for r in answerable])

    confidences = [r["confidence"] for r in rows]
    should_abstain = [not r["answerable"] for r in rows]
    best = pick_threshold(confidences, should_abstain)
    abstention = {
        "threshold": best["threshold"], "precision": best["precision"], "recall": best["recall"],
        "f1": best["f1"], "answered_rate": best["answered_rate"],
        "cv_f1": cross_validated_f1(confidences, should_abstain),
        "auc": roc_auc(confidences, should_abstain),
    }
    # End-to-end view at the chosen threshold: answerable questions that are
    # both kept (not abstained) AND have the evidence in the top 4 chunks.
    kept_and_found = sum(r["confidence"] >= best["threshold"] and r["rank"] is not None
                         and r["rank"] <= CHAT_K for r in answerable)
    metrics["answered_with_evidence"] = kept_and_found / len(answerable)

    return {
        "config": config, "metrics": metrics, "abstention": abstention,
        "pr_curve": precision_recall_curve(confidences, should_abstain),
        "index_seconds": index_seconds,
        "ms_per_query": 1000 * sum(latencies) / len(latencies),
        "per_question": rows,
    }


def add_paired_diffs(results, baseline="baseline-minilm"):
    """MRR difference vs the baseline on the same questions, with a 95% CI."""
    base = next((r for r in results if r["config"] == baseline), None)
    if base is None:
        return
    base_rr = {row["id"]: row["rr"] for row in base["per_question"] if row["answerable"]}
    for r in results:
        rows = [row for row in r["per_question"] if row["answerable"]]
        r["metrics"]["mrr_diff_vs_baseline"] = paired_bootstrap_diff(
            [row["rr"] for row in rows], [base_rr[row["id"]] for row in rows])


def results_table(results):
    lines = [
        "| Config | Hit@1 | Hit@4 | Hit@10 | MRR@10 [95% CI] | ΔMRR vs baseline [95% CI] "
        "| Abstain F1 (in-sample / 5-fold CV) | AUC | Answered w/ evidence | ms/query |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, a = r["metrics"], r["abstention"]
        lo, hi = m["mrr@10_ci"]
        diff = m.get("mrr_diff_vs_baseline")
        diff_cell = "—" if diff is None or r["config"] == "baseline-minilm" else \
            f"{diff[0]:+.2f} [{diff[1]:+.2f}, {diff[2]:+.2f}]"
        lines.append(
            f"| {r['config']} | {m['hit@1']:.2f} | {m['hit@4']:.2f} | {m['hit@10']:.2f} "
            f"| {m['mrr@10']:.2f} [{lo:.2f}, {hi:.2f}] | {diff_cell} "
            f"| {a['f1']:.2f} / {a['cv_f1']:.2f} | {a['auc']:.2f} "
            f"| {m['answered_with_evidence']:.2f} | {r['ms_per_query']:.0f} |"
        )
    return "\n".join(lines)


def plot_pr_curves(results, path, show=("baseline-minilm", "hybrid-bge-small")):
    """Abstention precision-recall curves for the baseline, hybrid, and best config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = max(results, key=lambda r: (r["metrics"]["mrr@10"], r["abstention"]["auc"]))["config"]
    wanted = list(dict.fromkeys([*show, best]))
    chosen = [r for r in results if r["config"] in wanted]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three validated categorical slots

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for color, r in zip(colors, chosen):
        curve = sorted(r["pr_curve"], key=lambda p: p["recall"])
        ax.plot([p["recall"] for p in curve], [p["precision"] for p in curve],
                color=color, linewidth=2, label=f"{r['config']} (AUC {r['abstention']['auc']:.2f})")
        a = r["abstention"]
        ax.scatter([a["recall"]], [a["precision"]], s=70, color=color,
                   edgecolor="#fcfcfb", linewidth=2, zorder=3)
        ax.annotate(f"t={a['threshold']:.2f}", (a["recall"], a["precision"]),
                    textcoords="offset points", xytext=(6, -14), fontsize=8, color="#52514e")
    ax.set_xlabel("Recall: share of unanswerable questions we abstain on", color="#52514e")
    ax.set_ylabel("Precision: share of abstentions that were right", color="#52514e")
    ax.set_title("Abstention precision-recall (dot = best-F1 threshold)", color="#0b0b0b", loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_labels_review(questions, chunks, relevant, path):
    lines = [
        "# Evaluation labels - please review",
        "",
        "Each answerable question lists the evidence quote(s) that define a correct",
        "retrieval, and the chunks/pages where the harness actually found that",
        "evidence. Check that (1) the answer is right, (2) the evidence supports it,",
        "(3) unanswerable questions really have no answer in the corpus.",
        "",
        "Labels were written from the label text first, then expanded by pooling: the",
        "top-ranked chunk from every retrieval config was judged by hand, and chunks",
        "that answer the question in different words (e.g. the Patient Information",
        "version of a Highlights fact) got their wording added as extra evidence.",
        "",
        "| ID | Question | Source | Expected answer | Evidence quote(s) | Found on pages (chunks) |",
        "|---|---|---|---|---|---|",
    ]
    for q in questions:
        if q["answerable"]:
            ids = sorted(relevant[q["id"]])
            pages = sorted({p for i in ids for p in range(chunks[i]["page_start"], chunks[i]["page_end"] + 1)})
            evidence = "<br>".join(f"“{e}”" for e in q["evidence"])
            lines.append(f"| {q['id']} | {q['question']} | {q['source_file']} | {q['answer']} "
                         f"| {evidence} | p.{', '.join(map(str, pages))} ({len(ids)}) |")
        else:
            lines.append(f"| {q['id']} | {q['question']} | — | **UNANSWERABLE** ({q['why_unanswerable']}) | — | — |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval configs on the medical eval set.")
    parser.add_argument("--configs", nargs="+", default=list(RETRIEVER_CONFIGS), choices=list(RETRIEVER_CONFIGS))
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text())
    chunks = load_corpus_chunks()
    relevant = label_relevance(questions, chunks)
    RESULTS_DIR.mkdir(exist_ok=True)
    write_labels_review(questions, chunks, relevant, RESULTS_DIR / "labels_review.md")
    print(f"{len(chunks)} chunks, {len(questions)} questions "
          f"({sum(not q['answerable'] for q in questions)} unanswerable). Labels OK.\n")

    results = []
    for config in args.configs:
        print(f"== {config}")
        results.append(evaluate_config(config, chunks, questions, relevant))
        m, a = results[-1]["metrics"], results[-1]["abstention"]
        print(f"   Hit@1={m['hit@1']:.2f} Hit@4={m['hit@4']:.2f} MRR@10={m['mrr@10']:.2f} "
              f"abstainF1={a['f1']:.2f} (cv {a['cv_f1']:.2f}) AUC={a['auc']:.2f}")

    add_paired_diffs(results)
    table = results_table(results)
    (RESULTS_DIR / "results.md").write_text(
        f"Corpus: {len(chunks)} chunks from {len(list(CORPUS_DIR.glob('*.pdf')))} PDFs. "
        f"Questions: {len(questions)} ({sum(not q['answerable'] for q in questions)} unanswerable).\n\n"
        f"{table}\n"
    )
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    plot_pr_curves(results, RESULTS_DIR / "pr_curves.png")
    print(f"\n{table}\n\nWrote {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
