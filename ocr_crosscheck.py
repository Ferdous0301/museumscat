"""
Step 1.5 (optional, run after run_inference.py, before risk_ranking.py):
cross-check VLM predictions against an independent OCR engine (Tesseract)
reading the SAME preprocessed pixels.

WHY: Qwen2.5-VL is a general vision-language model, not a specialized OCR
engine. Character-level confusions (the "22.5.1977." -> "22.S.1977." case)
are exactly the kind of error a dedicated OCR engine, despite having zero
understanding of what a "date" or "locality" even is, is often better at,
because that's literally what it's built and trained for. This script does
NOT try to replace the VLM's judgment about which text is the date/locality
vs. metadata -- that's a semantic decision Tesseract has no way to make.
It only:
    1. produces an "agreement" score (0-1) per field: how well the VLM's
       answer matches something Tesseract also read from the same image.
       This is a strong feature for risk_ranking.py, since it comes from a
       genuinely different architecture rather than the VLM checking itself.
    2. applies ONE narrow, high-precision repair: if a Tesseract reading is
       within 1 edit of the VLM's date and that edit swaps a known
       confusable character pair (5/S, 0/O, 1/l/I, 8/B) in a way that makes
       the date MORE digit-heavy (dates should be mostly digits), prefer the
       Tesseract version. Locality is NEVER auto-repaired this way -- word
       shapes are too ambiguous for a blind character-swap heuristic to be
       safe there, unlike a date where "should mostly be digits" is a
       strong, checkable prior.

INSTALL (Kaggle):
    !apt-get -qq install -y tesseract-ocr tesseract-ocr-dan
    !pip install -q pytesseract
    (falls back to English-only OCR with a warning if the Danish pack isn't
    installed -- still useful for digit-heavy date fields, less useful for
    locality's Danish characters æ/ø/å)

Usage:
    python ocr_crosscheck.py \
        --preds train_preds_raw.csv \
        --images /kaggle/input/.../images \
        --out train_preds_ocr.csv \
        [--lang dan+eng]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib

from image_utils import smart_resize_image, TARGET_MIN_PIXELS, TARGET_MAX_PIXELS


# ================================================================
# CONFUSABLE CHARACTER PAIRS (date-field repair only)
# ================================================================

# Each pair is considered interchangeable for repair purposes. Order
# doesn't matter -- both directions are checked.
CONFUSABLE_PAIRS = [
    ("5", "S"), ("5", "s"),
    ("0", "O"), ("0", "o"),
    ("1", "l"), ("1", "I"), ("1", "i"),
    ("8", "B"),
    ("2", "Z"), ("2", "z"),
    ("6", "G"), ("6", "b"),
]
_CONFUSABLE_SET = {frozenset(p) for p in CONFUSABLE_PAIRS}

# A "date-like" token: digits, dots, slashes, hyphens, spaces, and roman
# numerals, at least 3 characters, used to pull date candidates out of
# Tesseract's raw (unstructured, multi-line) output. Trailing period is
# optional but explicitly captured -- dates like "22.5.1977." end with one,
# and without this the token would be extracted one character short,
# causing a spurious length mismatch against the VLM's (correctly full-
# length) prediction in the confusable-swap check below.
_DATE_TOKEN_RE = re.compile(
    r"[0-9IVXLCDM][0-9IVXLCDMivxlcdm.\-/ ]{2,}[0-9IVXLCDM]\.?"
)


# ================================================================
# OCR
# ================================================================

def run_ocr(image: Image.Image, lang: str) -> str:
    if not _HAS_TESSERACT:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract "
            "and apt-get install tesseract-ocr tesseract-ocr-dan"
        )
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractError as e:
        # Common cause: requested a language pack (e.g. 'dan') that isn't
        # installed. Fall back to English-only rather than crashing the
        # whole run over one image.
        if "dan" in lang:
            print(f"    [OCR language fallback: '{lang}' failed ({e}), retrying with 'eng']")
            return pytesseract.image_to_string(image, lang="eng")
        raise


# ================================================================
# AGREEMENT SCORING
# ================================================================

def agreement_score(value: str, ocr_text: str) -> float:
    """
    How well `value` (the VLM's prediction) matches something present
    anywhere in `ocr_text` (Tesseract's raw, unstructured multi-line
    output for the whole image). Returns 0.5 (neutral/unknown) for
    MISSING values, since there's nothing to check agreement against --
    OCR finding text elsewhere on a busy label doesn't confirm or deny
    that THIS field is genuinely absent.
    """
    if not isinstance(value, str) or value.strip().upper() in ("MISSING", ""):
        return 0.5
    if not ocr_text or not ocr_text.strip():
        return 0.0

    # Multi-card values: score each segment against the OCR text and take
    # the best, then average -- a value is only as trustworthy as its
    # least-supported segment, but one missing OCR line for a genuinely
    # faint second card shouldn't zero out an otherwise well-supported
    # first card.
    segments = [s.strip() for s in value.split("|") if s.strip()]
    if not segments:
        return 0.5

    scores = []
    for seg in segments:
        if _HAS_RAPIDFUZZ:
            # partial_ratio finds the best-aligned substring of ocr_text
            # against seg, which is exactly what we want for "does this
            # short field value appear somewhere in this long OCR blob".
            score = _rf_fuzz.partial_ratio(seg.lower(), ocr_text.lower()) / 100.0
        else:
            score = _best_partial_ratio_fallback(seg.lower(), ocr_text.lower())
        scores.append(score)

    return sum(scores) / len(scores)


def _best_partial_ratio_fallback(needle: str, haystack: str) -> float:
    """
    difflib-based fallback for partial (substring) matching when rapidfuzz
    isn't installed. Slides a window of len(needle) (+/- 2 chars) across
    haystack. Haystack is OCR output for one label image, so this is at
    most a few hundred characters -- fine to brute-force.
    """
    n = len(needle)
    if n == 0 or len(haystack) == 0:
        return 0.0

    best = 0.0
    for wlen in range(max(1, n - 2), n + 3):
        step = max(1, wlen // 4)  # coarser stride keeps this fast
        for start in range(0, max(1, len(haystack) - wlen + 1), step):
            window = haystack[start:start + wlen]
            score = difflib.SequenceMatcher(None, needle, window).ratio()
            if score > best:
                best = score
    return best


# ================================================================
# DATE-FIELD REPAIR
# ================================================================

def _differs_by_confusable_swap(a: str, b: str) -> bool:
    """True if a and b are the same length and differ in exactly one
    position, where that position's characters form a known confusable
    pair."""
    if len(a) != len(b):
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) != 1:
        return False
    x, y = diffs[0]
    return frozenset((x, y)) in _CONFUSABLE_SET


def _digit_count(s: str) -> int:
    return sum(c.isdigit() for c in s)


def repair_date_with_ocr(vlm_date: str, ocr_text: str):
    """
    Returns (repaired_date_or_original, was_repaired: bool).

    Only fires when a candidate substring from the OCR text differs from
    the VLM's date by exactly one confusable-character swap AND the OCR
    candidate has MORE digits than the VLM version (dates should be
    digit-heavy, so this is evidence the OCR reading is the more likely
    correct one, not just a different-but-equally-plausible reading).
    """
    if not isinstance(vlm_date, str) or vlm_date.strip().upper() in ("MISSING", ""):
        return vlm_date, False
    if not ocr_text or not ocr_text.strip():
        return vlm_date, False

    candidates = _DATE_TOKEN_RE.findall(ocr_text)
    for cand in candidates:
        cand = cand.strip()
        if _differs_by_confusable_swap(vlm_date, cand) and _digit_count(cand) > _digit_count(vlm_date):
            return cand, True

    return vlm_date, False


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="predictions CSV from run_inference.py")
    ap.add_argument("--images", required=True, help="directory containing the images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="dan+eng",
                     help="Tesseract language(s). Falls back to 'eng' automatically "
                          "if the Danish pack isn't installed.")
    ap.add_argument("--apply-date-repair", action="store_true",
                     help="If set, overwrite verbatimDate in the output CSV when the "
                          "confusable-character repair fires. Off by default -- the "
                          "repair candidate and agreement scores are always written as "
                          "separate columns regardless, so you can inspect before "
                          "deciding to apply it, or feed it into risk_ranking.py instead.")
    args = ap.parse_args()

    if not _HAS_TESSERACT:
        raise RuntimeError(
            "pytesseract not installed. On Kaggle:\n"
            "  !apt-get -qq install -y tesseract-ocr tesseract-ocr-dan\n"
            "  !pip install -q pytesseract"
        )

    preds = pd.read_csv(args.preds)
    images_dir = Path(args.images)

    print(f"OCR backend: pytesseract, lang={args.lang}")
    print(f"Substring matching: {'rapidfuzz' if _HAS_RAPIDFUZZ else 'difflib fallback'}")

    date_agree, loc_agree = [], []
    date_repaired_vals, date_repaired_flags = [], []

    for i, row in preds.iterrows():
        image_path = images_dir / str(row["image_file"])

        if not image_path.exists():
            print(f"[{i + 1}/{len(preds)}] {row['image_file']} image not found, skipping OCR")
            date_agree.append(0.5)
            loc_agree.append(0.5)
            date_repaired_vals.append(row["verbatimDate"])
            date_repaired_flags.append(False)
            continue

        try:
            img = smart_resize_image(image_path, min_pixels=TARGET_MIN_PIXELS, max_pixels=TARGET_MAX_PIXELS)
            ocr_text = run_ocr(img, args.lang)
        except Exception as e:
            print(f"[{i + 1}/{len(preds)}] {row['image_file']} OCR FAILED: {e}")
            date_agree.append(0.5)
            loc_agree.append(0.5)
            date_repaired_vals.append(row["verbatimDate"])
            date_repaired_flags.append(False)
            continue

        d_agree = agreement_score(str(row["verbatimDate"]), ocr_text)
        l_agree = agreement_score(str(row["verbatimLocality"]), ocr_text)
        date_agree.append(d_agree)
        loc_agree.append(l_agree)

        repaired, was_repaired = repair_date_with_ocr(str(row["verbatimDate"]), ocr_text)
        date_repaired_vals.append(repaired)
        date_repaired_flags.append(was_repaired)

        note = f" [date repair: {row['verbatimDate']!r} -> {repaired!r}]" if was_repaired else ""
        print(f"[{i + 1}/{len(preds)}] {row['image_file']} -> "
              f"date_agree={d_agree:.2f} loc_agree={l_agree:.2f}{note}")

    preds["date_ocr_agreement"] = date_agree
    preds["locality_ocr_agreement"] = loc_agree
    preds["date_ocr_repair_candidate"] = date_repaired_vals
    preds["date_ocr_repaired"] = date_repaired_flags

    if args.apply_date_repair:
        n_applied = sum(date_repaired_flags)
        preds["verbatimDate_original"] = preds["verbatimDate"]
        preds["verbatimDate"] = date_repaired_vals
        print(f"Applied OCR date repair to {n_applied}/{len(preds)} rows.")

    preds.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()