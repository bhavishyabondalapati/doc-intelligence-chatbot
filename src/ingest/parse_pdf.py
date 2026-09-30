"""
Step 1: PDF Parsing & Layout-Aware Text Extraction
----------------------------------------------------
This is the local replacement for AWS Textract in the original project.

For every PDF in data/raw_pdfs/, this script:

  1. Tries to pull text directly out of the PDF (fast, works for
     "born-digital" pages - i.e. pages that already contain real,
     selectable text, like typed notes or a Word doc exported to PDF).

  2. If a page has little or no extractable text (common for scanned
     pages, or photographed textbook pages), it falls back to OCR:
     the page is rendered as an image, then Tesseract "reads" the
     image and returns text + where it found each word.

  3. For every piece of text found, we record WHERE on the page it
     sits - a "bounding box" (x0, y0, x1, y1). This is what "layout-
     aware" means: we're not just grabbing a blob of text, we're
     keeping track of position, so later steps can reason about
     structure (headings vs. body text vs. tables, etc).

Output: one JSON file per PDF in data/processed/, containing a list
of "blocks" - each with its text, page number, bounding box, and
which method (digital vs OCR) extracted it.

Run:
    python src/ingest/parse_pdf.py
"""

import json
from pathlib import Path

import fitz  # PyMuPDF - reads PDF structure directly
import pytesseract
from pdf2image import convert_from_path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_pdfs"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# If a page has fewer than this many characters of extractable text,
# we assume it's a scanned/image page and OCR it instead of trusting
# the (near-empty) digital text layer.
MIN_CHARS_FOR_DIGITAL_TEXT = 20


def extract_digital_text_blocks(page, page_number, source_file):
    """Pull text + position directly from a born-digital PDF page."""
    blocks = []
    # "blocks" mode returns each visual block as (x0, y0, x1, y1, text, ...)
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        text = text.strip()
        if not text:
            continue
        blocks.append({
            "text": text,
            "page": page_number,
            "bbox": {"x0": round(x0, 2), "y0": round(y0, 2),
                     "x1": round(x1, 2), "y1": round(y1, 2)},
            "source_file": source_file,
            "extraction_method": "digital",
        })
    return blocks


def extract_ocr_text_blocks(pil_image, page_number, source_file):
    """
    Run Tesseract OCR on a page image and group detected words into
    lines (using Tesseract's own block/paragraph/line numbering),
    keeping a merged bounding box per line.
    """
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    blocks = []
    current_line, current_bbox, last_line_key = [], None, None

    def flush():
        if current_line:
            blocks.append({
                "text": " ".join(current_line),
                "page": page_number,
                "bbox": current_bbox,
                "source_file": source_file,
                "extraction_method": "ocr",
            })

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word:
            continue

        line_key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        if line_key != last_line_key and current_line:
            flush()
            current_line, current_bbox = [], None

        current_line.append(word)
        if current_bbox is None:
            current_bbox = {"x0": x, "y0": y, "x1": x + w, "y1": y + h}
        else:
            current_bbox["x0"] = min(current_bbox["x0"], x)
            current_bbox["y0"] = min(current_bbox["y0"], y)
            current_bbox["x1"] = max(current_bbox["x1"], x + w)
            current_bbox["y1"] = max(current_bbox["y1"], y + h)

        last_line_key = line_key

    flush()
    return blocks


def process_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    ocr_page_count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        digital_text = page.get_text().strip()

        if len(digital_text) >= MIN_CHARS_FOR_DIGITAL_TEXT:
            blocks = extract_digital_text_blocks(page, page_number, pdf_path.name)
        else:
            ocr_page_count += 1
            images = convert_from_path(
                str(pdf_path), first_page=page_number, last_page=page_number, dpi=200
            )
            blocks = extract_ocr_text_blocks(images[0], page_number, pdf_path.name)

        all_blocks.extend(blocks)

    doc.close()
    return all_blocks, ocr_page_count


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        print("Drop a few of your own PDFs there, then re-run this script.")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        blocks, ocr_pages = process_pdf(pdf_path)

        out_path = OUT_DIR / f"{pdf_path.stem}.json"
        with open(out_path, "w") as f:
            json.dump(blocks, f, indent=2)

        digital_count = sum(1 for b in blocks if b["extraction_method"] == "digital")
        ocr_count = sum(1 for b in blocks if b["extraction_method"] == "ocr")

        print(f"  -> {len(blocks)} text blocks "
              f"({digital_count} digital, {ocr_count} OCR)")
        print(f"  -> {ocr_pages} page(s) needed OCR fallback")
        print(f"  -> Saved to {out_path}\n")

    print("Done. Open a file in data/processed/ to see the extracted blocks.")


if __name__ == "__main__":
    main()
