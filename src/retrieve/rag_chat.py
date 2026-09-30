"""
Step 3: RAG Chat (Retrieval-Augmented Generation) - Gemini version
----------------------------------------------------------------------
This is where retrieval becomes an actual chatbot. For each question:

  1. Embed the question and search FAISS for the most relevant chunks
     (exactly what search_demo.py did in Step 2c).
  2. Stitch those chunks together into a "context" block.
  3. Send the question + context to Google's Gemini API, with strict
     instructions to answer ONLY using the provided context - and to
     say so plainly if the context doesn't contain the answer.

That last part matters a lot. Without it, an LLM will often answer
confidently even when it's making things up ("hallucinating"), because
that's what language models are trained to do by default: produce
plausible-sounding text. Grounding it in retrieved context, and telling
it explicitly to admit uncertainty, is what makes a RAG chatbot
trustworthy instead of just persuasive.

We also print the top retrieval score alongside the answer - the same
confidence signal from Step 2c. If it's low, be skeptical of the answer
even if it *sounds* confident.

Setup:
  1. Copy .env.example to .env
  2. Put your real Gemini API key in .env (never commit this file -
     it's already listed in .gitignore)

Run:
    python src/retrieve/rag_chat.py
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads GEMINI_API_KEY from a local .env file

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHAT_MODEL = "gemini-flash-latest"  # auto-updating alias - always points to
                                     # Google's current stable Flash model,
                                     # so this won't break again when a
                                     # specific version gets retired
TOP_K = 4
LOW_CONFIDENCE_THRESHOLD = 0.35  # tune based on what you saw in Step 2c

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using \
ONLY the context provided below. If the context does not contain enough \
information to answer confidently, say so plainly instead of guessing. \
Always mention which source file (and page, if given) your answer came from."""


MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15


def generate_with_retry(client, **kwargs):
    """
    The Gemini API can return a transient 503 when a model is under heavy
    load - that's the server being busy, not a bug in our code. We retry
    a few times with a short wait before giving up, instead of crashing
    on the first hiccup. This is standard practice for any app that talks
    to an external API.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.ServerError:
            if attempt == MAX_RETRIES:
                raise
            print(f"[Model busy (attempt {attempt}/{MAX_RETRIES}) - "
                  f"retrying in {RETRY_DELAY_SECONDS}s...]")
            time.sleep(RETRY_DELAY_SECONDS)


def build_context(chunks_with_scores):
    parts = []
    for chunk, score in chunks_with_scores:
        parts.append(
            f"[Source: {chunk['source_file']}, p.{chunk['page_start']}-{chunk['page_end']}, "
            f"relevance={score:.2f}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("No GEMINI_API_KEY found. Copy .env.example to .env and add your key.")
        return

    index_path = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.json"
    if not index_path.exists():
        print("No index found. Run src/embed/build_index.py first.")
        return

    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = genai.Client()  # reads GEMINI_API_KEY from the environment

    print(f"\nLoaded {index.ntotal} chunks. Ask a question about your documents (or 'quit').\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        query_vector = embed_model.encode([question], normalize_embeddings=True)
        query_vector = np.array(query_vector, dtype="float32")
        scores, indices = index.search(query_vector, TOP_K)

        retrieved = [(metadata[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
        top_score = retrieved[0][1]

        if top_score < LOW_CONFIDENCE_THRESHOLD:
            print(f"\n[Retrieval confidence is low (top score={top_score:.2f}) - "
                  f"treat the answer below with skepticism.]")

        context = build_context(retrieved)

        try:
            response = generate_with_retry(
                client,
                model=CHAT_MODEL,
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0,
                ),
            )
        except genai_errors.ServerError:
            print("\n[Gemini's servers are still overloaded after retries. "
                  "Wait a bit and try this question again.]\n")
            continue

        print(f"\nBot: {response.text}\n")


if __name__ == "__main__":
    main()
