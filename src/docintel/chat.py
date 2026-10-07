"""
RAG Chat (Retrieval-Augmented Generation) with Google Gemini
-------------------------------------------------------------
For each question:

  1. Retrieve the most relevant chunks (see retrieval.py).
  2. If retrieval confidence is below the threshold, abstain: say "I don't
     have enough information" without calling the LLM at all. The
     threshold comes from the precision-recall curve in eval/ - it is
     not a guess.
  3. Otherwise, send the question + chunks to Gemini with strict
     instructions to answer ONLY from that context, and to say so
     plainly if the context doesn't contain the answer.
  4. Print the sources from retrieval itself (file, page, and the
     bounding box of the first region), so citations don't depend on the
     LLM remembering to include them.

Setup:
  1. Copy .env.example to .env
  2. Put your real Gemini API key in .env (never commit this file -
     it's already listed in .gitignore)

Run:
    python -m docintel.chat [--config hybrid-bge-small] [--threshold 0.702]
"""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from docintel.config import INDEX_DIR, PROJECT_ROOT
from docintel.retrieval import DEFAULT_CONFIG, RETRIEVER_CONFIGS
from docintel.search import load_retriever

CHAT_MODEL = "gemini-3.8-flash"  # pinned (not "-latest") so answers are reproducible
TOP_K = 4

# Abstention thresholds per retrieval config, picked by eval/run_eval.py as
# the best-F1 point on the precision-recall curve (eval/results/). On the
# eval set, 0.702 for the default config abstained on 7/12 unanswerable
# questions (87.5% precision) while still answering 47/48 answerable ones.
# They are calibrated on drug labels: re-run the eval on your own documents.
ABSTAIN_THRESHOLDS = {
    "hybrid-bge-small": 0.702,
    "dense-bge-small": 0.702,
    "baseline-minilm": 0.629,
}

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using \
ONLY the context provided below. If the context does not contain enough \
information to answer confidently, say "I don't have enough information" \
instead of guessing. Always mention which source file and page your answer \
came from."""

MAX_RETRIES = 4
BASE_DELAY_SECONDS = 2


def generate_with_retry(client, **kwargs):
    """
    Gemini can return 503 (overloaded) or 429 (rate limited). Both are
    temporary, so we retry with exponential backoff - waiting 2s, 4s, 8s -
    instead of crashing on the first hiccup. Other errors (bad key, bad
    request) are our fault and retrying won't help, so they raise at once.
    """
    from google.genai import errors as genai_errors

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as err:
            retryable = isinstance(err, genai_errors.ServerError) or err.code == 429
            if not retryable or attempt == MAX_RETRIES:
                raise
            delay = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
            print(f"[Gemini error {err.code} (attempt {attempt}/{MAX_RETRIES}) - retrying in {delay}s]")
            time.sleep(delay)


def build_context(chunks, hits):
    parts = []
    for idx, score in hits:
        chunk = chunks[idx]
        parts.append(f"[Source: {chunk['source_file']}, p.{chunk['page_start']}-{chunk['page_end']}]"
                     f"\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def format_sources(chunks, hits):
    lines = []
    for idx, score in hits:
        chunk = chunks[idx]
        region = chunk.get("regions", [{}])[0]
        box = region.get("bbox")
        where = f", bbox=({box['x0']:.0f},{box['y0']:.0f},{box['x1']:.0f},{box['y1']:.0f})" if box else ""
        lines.append(f"  - {chunk['source_file']} p.{chunk['page_start']}-{chunk['page_end']}"
                     f"{where} [rank score={score:.3f}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Chat with your PDFs.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG, choices=sorted(RETRIEVER_CONFIGS))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Abstain below this retrieval confidence (default: eval-picked value)")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")  # reads GEMINI_API_KEY from the project's .env file
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("No GEMINI_API_KEY found. Copy .env.example to .env and add your key.")

    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    threshold = args.threshold if args.threshold is not None else ABSTAIN_THRESHOLDS.get(args.config)
    retriever, chunks = load_retriever(args.index_dir, args.config)
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    print(f"\nLoaded {len(chunks)} chunks ({args.config}, abstain below {threshold}). "
          f"Ask a question (or 'quit').\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        result = retriever.search(question, TOP_K)
        if threshold is not None and result.confidence < threshold:
            print(f"\nBot: I don't have enough information in these documents to answer that. "
                  f"(retrieval confidence {result.confidence:.3f} < {threshold})\n")
            continue

        try:
            response = generate_with_retry(
                client,
                model=CHAT_MODEL,
                contents=f"Context:\n{build_context(chunks, result.hits)}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=0,
                    # we don't give Gemini any tools, so switch off tool calling
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except genai_errors.APIError as err:
            print(f"\n[Gemini request failed: {err}]\n")
            continue

        print(f"\nBot: {response.text}\n\nRetrieved sources (confidence={result.confidence:.3f}):\n"
              f"{format_sources(chunks, result.hits)}\n")


if __name__ == "__main__":
    main()
