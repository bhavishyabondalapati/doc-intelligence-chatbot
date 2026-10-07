"""
Step 1: PDF Parsing & Layout-Aware Text Extraction
----------------------------------------------------
For every PDF in a folder, this script:

  1. Tries to pull text directly out of the PDF (fast, works for
     "born-digital" pages - i.e. pages that already contain real,
     selectable text, like a Word doc exported to PDF).

  2. If a page has little or no extractable text (scanned pages,
     photographed pages), it falls back to OCR: the page is rendered as
     an image, then Tesseract "reads" the image and returns text + where
     it found each word.

  3. For every piece of text found, we record WHERE on the page it
     sits - a "bounding box" (x0, y0, x1, y1). Both paths report boxes
     in the same unit: PDF points (1/72 inch, origin top-left), so a box
     from an OCR'd page and a box from a digital page can be compared or
     drawn on the page the same way.

Output: one JSON file per PDF, containing a list of "blocks" - each with
its text, page number, bounding box, and which method (digital vs OCR)
extracted it.

Run:
    python -m docintel.ingest [--pdf-dir data/raw_pdfs] [--out-dir data/processed]
"""

import argparse
import json
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from docintel.config import PROCESSED_DIR, RAW_PDF_DIR

# If a page has fewer than this many characters of extractable text,
# we assume it's a scanned/image page and OCR it instead of trusting
# the (near-empty) digital text layer.
MIN_CHARS_FOR_DIGITAL_TEXT = 20

# Resolution used to render a page for OCR. 200 DPI is a common
# accuracy/speed sweet spot for Tesseract.
OCR_DPI = 200


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


def extract_ocr_text_blocks(pil_image, page_number, source_file, dpi=OCR_DPI):
    """
    Run Tesseract OCR on a page image and group detected words into
    lines (using Tesseract's own block/paragraph/line numbering),
    keeping a merged bounding box per line.

    Tesseract reports pixel coordinates of the rendered image, so we
    divide by (dpi / 72) to convert them to PDF points - the same unit
    the digital path uses.
    """
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    scale = 72.0 / dpi
    lines = {}  # (block, par, line) -> {"words": [...], "bbox": [...]}, in reading order

    for i, raw_word in enumerate(data["text"]):
        word = raw_word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        line = lines.setdefault(key, {"words": [], "bbox": [x, y, x + w, y + h]})
        line["words"].append(word)
        box = line["bbox"]
        line["bbox"] = [min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + h)]

    return [
        {
            "text": " ".join(line["words"]),
            "page": page_number,
            "bbox": {k: round(v * scale, 2) for k, v in zip(("x0", "y0", "x1", "y1"), line["bbox"])},
            "source_file": source_file,
            "extraction_method": "ocr",
        }
        for line in lines.values()
    ]


def render_page(page, dpi=OCR_DPI):
    """Render one PDF page to a PIL image (PyMuPDF does this - no poppler needed)."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def process_pdf(pdf_path: Path):
    """Return (blocks, number_of_pages_that_needed_ocr) for one PDF."""
    doc = pymupdf.open(pdf_path)
    all_blocks = []
    ocr_page_count = 0

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        if len(page.get_text().strip()) >= MIN_CHARS_FOR_DIGITAL_TEXT:
            blocks = extract_digital_text_blocks(page, page_number, pdf_path.name)
        else:
            ocr_page_count += 1
            blocks = extract_ocr_text_blocks(render_page(page), page_number, pdf_path.name)
        all_blocks.extend(blocks)

    doc.close()
    return all_blocks, ocr_page_count


def ingest_folder(pdf_dir: Path, out_dir: Path, verbose=True):
    """Parse every PDF in pdf_dir and write one blocks JSON per PDF to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files and verbose:
        print(f"No PDFs found in {pdf_dir}. Drop a few PDFs there, then re-run.")

    for pdf_path in pdf_files:
        blocks, ocr_pages = process_pdf(pdf_path)
        out_path = out_dir / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(blocks, indent=2))
        if verbose:
            ocr_count = sum(b["extraction_method"] == "ocr" for b in blocks)
            print(f"{pdf_path.name}: {len(blocks)} blocks ({ocr_count} from OCR, "
                  f"{ocr_pages} page(s) needed OCR) -> {out_path}")
    return pdf_files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--pdf-dir", type=Path, default=RAW_PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    ingest_folder(args.pdf_dir, args.out_dir)


if __name__ == "__main__":
    main()
