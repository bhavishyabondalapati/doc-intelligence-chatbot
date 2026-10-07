"""
Make a *simulated* scanned PDF for the eval corpus.

Real scanned FDA documents exist, but the FDA archive blocks scripted
downloads, so instead we take the first pages of a public DailyMed label
(amoxicillin capsules) and turn them into what a cheap office scan looks
like: rendered at 150 DPI, grayscale, rotated about half a degree, a bit
of blur and sensor noise, JPEG-compressed, saved as image-only pages.
There is no text layer, so the pipeline has to OCR it.

Run (from the project root):
    python eval/make_scanned_pdf.py
"""

import io
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

EVAL_DIR = Path(__file__).resolve().parent
SOURCE = EVAL_DIR / "source_pdfs" / "amoxicillin_capsule.pdf"
OUTPUT = EVAL_DIR / "corpus" / "amoxicillin_capsule_SCANNED.pdf"
PAGES = 4
DPI = 150


def degrade(img, rng, angle):
    img = img.convert("L").rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    noisy = np.asarray(img, dtype=np.float32) + rng.normal(0, 8, size=(img.height, img.width))
    img = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=60)
    return buffer.getvalue()


def main():
    rng = np.random.default_rng(42)
    src = pymupdf.open(SOURCE)
    out = pymupdf.open()
    for page_index in range(PAGES):
        page = src[page_index]
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        jpeg = degrade(img, rng, angle=rng.uniform(-0.7, 0.7))
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=jpeg)
    out.save(OUTPUT, deflate=True)
    print(f"Wrote {OUTPUT} ({PAGES} image-only pages, {OUTPUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
