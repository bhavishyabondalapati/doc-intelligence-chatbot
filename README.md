# Document Intelligence Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot that turns a folder of
PDFs into a searchable, source-grounded Q&A system — built from the ground
up (no framework doing the heavy lifting) to understand and demonstrate
every layer of a real GenAI pipeline.

## What it does

Point it at a folder of PDFs. It will:

1. **Parse** them — direct text extraction for digital PDFs, automatic OCR
   fallback (Tesseract) for scanned/image pages, preserving on-page
   position metadata
2. **Chunk** the extracted text into overlapping segments, sized to keep
   each chunk semantically focused
3. **Embed** each chunk into a vector locally (HuggingFace
   `sentence-transformers`, no cloud call needed for this step)
4. **Index** those vectors in FAISS for fast similarity search
5. On a question: **retrieve** the most relevant chunks, then **generate**
   a grounded answer (Google Gemini) — or explicitly say "I don't have
   enough information" when the retrieved context doesn't support a
   confident answer

## Notable engineering decisions

- **Confidence-aware retrieval** — every answer carries a similarity
  score; below a tunable threshold, the system flags its own answer as
  low-confidence instead of presenting it as fact.
- **Anti-hallucination prompting** — the LLM is instructed to answer only
  from the provided context and to say so plainly when it can't. Verified
  empirically: asking about something demonstrably absent from the source
  PDFs correctly returns "I don't have enough information," not a
  plausible-sounding guess.
- **Resilience to upstream failures** — automatic retry with backoff on
  transient API server errors, instead of crashing on the first hiccup.
- **Layout-aware parsing with OCR fallback** — chunks retain page and
  bounding-box metadata; scanned pages are detected automatically and
  routed through OCR without manual intervention.

## Architecture

```
PDFs
  -> Parse       (PyMuPDF, + Tesseract OCR fallback for scanned pages)
  -> Chunk       (overlapping ~120-word windows)
  -> Embed       (sentence-transformers: all-MiniLM-L6-v2)
  -> Index       (FAISS, cosine similarity)
  -> Retrieve    (top-k relevant chunks for a question)
  -> Generate    (Google Gemini, grounded + confidence-scored)
```

## Tech stack

Python · PyMuPDF · Tesseract OCR · sentence-transformers · FAISS ·
Google Gemini API

## Project structure

```
src/
  ingest/    PDF parsing (parse_pdf.py)
  embed/     Chunking + embeddings + FAISS index
  retrieve/  RAG chat: retrieval + grounded generation (rag_chat.py)
data/
  raw_pdfs/    input PDFs        (add your own - not tracked in git)
  processed/   parsed blocks     (generated)
  chunks/      chunked text      (generated)
  index/       FAISS index       (generated)
```

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install tesseract poppler      # macOS OCR/PDF system dependencies
cp .env.example .env                # then add your GEMINI_API_KEY
```

## Run the pipeline

```
python src/ingest/parse_pdf.py          # 1. parse PDFs in data/raw_pdfs/
python src/embed/chunk_documents.py     # 2. chunk the parsed text
python src/embed/build_index.py         # 3. embed + build FAISS index
python src/retrieve/rag_chat.py         # 4. ask questions
```

## Example interaction

```
You: what's the weight of the final project

Bot: Based on the provided context, the Final Project accounts for 40%
of the grade (aligned with learning objectives LO3-9). Students work in
groups of 3-4 and submit multiple deliverables throughout the semester.
Source: CS6983_Syllabus_Fall2026.pdf, p.2
```

## Note on the data folder

`data/raw_pdfs/` is empty in this repo — the original development and
testing set used personal documents that aren't included here for
privacy. Add your own PDFs to `data/raw_pdfs/` to try it out.