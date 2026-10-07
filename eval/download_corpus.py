"""
Re-download the eval corpus from DailyMed (U.S. National Library of Medicine).

The PDFs are committed in eval/corpus/ so results are reproducible:
DailyMed always serves the *latest* label version, and a label update
could move or reword the evidence the questions point to. Use this script
only to inspect the sources or refresh them deliberately (then re-check
the labels with `pytest tests/test_eval_labels.py`).

Run (from the project root):
    python eval/download_corpus.py            # downloads to eval/corpus_fresh/
"""

import urllib.request
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
URL = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={setid}"

# filename -> (DailyMed set ID, label version used, publish date)
LABELS = {
    "lipitor_atorvastatin.pdf": ("a60cc18b-0631-4cf0-b021-9f52224ece65", 8, "2026-07-15"),
    "zoloft_sertraline.pdf": ("fda754f6-d0f3-4dce-a17a-927d64f912f7", 7, "2025-07-24"),
    "zestril_lisinopril.pdf": ("838c2d78-d2d8-4981-9ec9-e50ef9e1a5d8", 2, "2025-05-03"),
    "jantoven_warfarin.pdf": ("19a69a72-ac5d-45d5-a94d-a5aaecbe4730", 13, "2025-07-10"),
    "metformin_laurus.pdf": ("c3dfa8a1-d10a-4a1a-8eba-5f4e2a5a2949", 8, "2026-09-26"),
    # source for the simulated scan (eval/make_scanned_pdf.py)
    "amoxicillin_capsule.pdf": ("9b3ab9ea-caee-4186-9068-43ca527c098d", 1, "2026-10-02"),
}


def main():
    out_dir = EVAL_DIR / "corpus_fresh"
    out_dir.mkdir(exist_ok=True)
    for filename, (setid, version, date) in LABELS.items():
        target = out_dir / filename
        urllib.request.urlretrieve(URL.format(setid=setid), target)
        print(f"{filename}: set {setid} (eval used v{version}, {date}) -> {target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
