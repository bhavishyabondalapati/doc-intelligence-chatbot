import shutil

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from docintel import ingest

HAS_TESSERACT = shutil.which("tesseract") is not None


def make_digital_pdf(path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Warfarin is an anticoagulant used to prevent clots.", fontsize=12)
    doc.save(path)


def make_scanned_pdf(path, dpi=200):
    """An image-only page (no text layer), like a scan."""
    width, height = int(8.5 * dpi), int(11 * dpi)
    img = Image.new("RGB", (width, height), "white")
    font = ImageFont.load_default(size=48)
    ImageDraw.Draw(img).text((200, 400), "Take one tablet daily with food", fill="black", font=font)
    img_path = path.with_suffix(".png")
    img.save(img_path)
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(img_path))
    doc.save(path)


def test_digital_page_gives_blocks_in_pdf_points(tmp_path):
    pdf = tmp_path / "digital.pdf"
    make_digital_pdf(pdf)
    blocks, ocr_pages = ingest.process_pdf(pdf)
    assert ocr_pages == 0
    assert "anticoagulant" in blocks[0]["text"]
    box = blocks[0]["bbox"]
    assert 60 < box["x0"] < 80 and box["x1"] <= 612 and box["y1"] <= 792


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")
def test_image_only_page_falls_back_to_ocr_with_bbox_in_points(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_scanned_pdf(pdf)
    blocks, ocr_pages = ingest.process_pdf(pdf)
    assert ocr_pages == 1
    assert all(b["extraction_method"] == "ocr" for b in blocks)
    assert "tablet daily" in " ".join(b["text"] for b in blocks).lower()
    # text was drawn at 200 px on a 200-DPI image = 72 pt; must be in points, not pixels
    assert 60 < blocks[0]["bbox"]["x0"] < 85
    assert blocks[0]["bbox"]["x1"] <= 612


def test_ocr_words_are_grouped_into_lines(monkeypatch):
    fake = {
        "text": ["Take", "one", "", "tablet", "Daily"],
        "block_num": [1, 1, 1, 1, 2], "par_num": [1, 1, 1, 1, 1], "line_num": [1, 1, 1, 2, 1],
        "left": [0, 100, 0, 0, 50], "top": [0, 0, 0, 50, 300],
        "width": [80, 60, 0, 120, 100], "height": [20, 20, 0, 20, 20],
    }
    monkeypatch.setattr(ingest.pytesseract, "image_to_data", lambda *a, **k: fake)
    blocks = ingest.extract_ocr_text_blocks(None, 3, "x.pdf", dpi=144)  # scale = 0.5
    assert [b["text"] for b in blocks] == ["Take one", "tablet", "Daily"]
    assert blocks[0]["bbox"] == {"x0": 0, "y0": 0, "x1": 80, "y1": 10}
    assert all(b["page"] == 3 for b in blocks)
