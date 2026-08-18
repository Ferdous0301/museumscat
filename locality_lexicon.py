"""
Step 2.5 (optional): Fuzzy-match predicted localities against a lexicon of known
Danish place names to correct spelling drift.

Why: exact-match-heavy metrics like NED reward precision. A VLM might transcribe
"Kongstrup" as "Kongstrop" or "Roenaes" instead of "Rønæs" — a fuzzy correction
against known place names can recover these cheaply, IF we're careful not to
"correct" a genuinely different (but real) place into the wrong one.

Two lexicon sources, combined:
1. Built from train.csv verbatimLocality values themselves (split on "|" for
   multi-card, then split on hierarchical separators). This captures the exact
   vocabulary/style used in this collection (abbreviations like "Kb", "V.", etc.)
   without needing any external data.
2. Optional: a user-supplied external gazetteer file (one place name per line,
   e.g. Danish parishes/towns) for broader coverage than 200 training rows.

We do NOT fuzzy-correct the whole locality string in one shot — hierarchical
localities like "V. Sjælland Røsnæs Kongstrup K1" mix multiple tokens (region,
parish, farm code) that shouldn't be merged. Instead we correct at the token/
segment level and only replace a segment when a lexicon entry is close enough
AND clearly closer than "no correction", so we don't overwrite ambiguous or
already-correct rare place names.

Usage:
    python locality_lexicon.py \
        --train-csv train.csv \
        --preds test_preds_raw.csv \
        --out test_preds_corrected.csv \
        [--gazetteer extra_places.txt] \
        [--threshold 0.82]
"""
import argparse
import re
from collections import Counter

import pandas as pd

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib

# Words that are locality-structural rather than place names — never treat these
# as candidates to fuzzy-match against, and never "correct" a token into these.
STOPWORDS = {
    "k1", "k2", "k3", "dania",  # "Dania" explicitly excluded per dataset description
}

SUBSTRATE_PHRASES = [
    r"\bi\s+kog[øo]dning\b", r"\bi\s+kok?j[øo]rning\b", r"\bkog[øo]dning\b",
]


def strip_substrate_phrases(s: str) -> str:
    for pat in SUBSTRATE_PHRASES:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", s).strip(" ,|")


def tokenize_locality(loc: str) -> list[str]:
    """Split a locality string into place-name segments. Splits on '|' (multi-card)
    and on common separators within a card (',', long dashes), but keeps
    multi-word place names ('Kulhuset Jægerspris') intact as single segments
    when they don't contain an internal separator."""
    if not isinstance(loc, str) or loc.strip().upper() in ("MISSING", "NULL", ""):
        return []
    segments = []
    for card in loc.split("|"):
        card = strip_substrate_phrases(card)
        for seg in re.split(r"[,]", card):
            seg = seg.strip(" .")
            if seg and seg.lower() not in STOPWORDS and seg.lower() != "dania":
                segments.append(seg)
    return segments


def build_lexicon_from_train(train_df: pd.DataFrame) -> Counter:
    """Build a frequency-weighted lexicon of known place segments from ground truth."""
    counter = Counter()
    for loc in train_df["verbatimLocality"].dropna():
        for seg in tokenize_locality(loc):
            counter[seg] += 1
    return counter


def load_external_gazetteer(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def best_match(segment: str, candidates: list[str], threshold: float):
    """Return (best_candidate, score) if a confident fuzzy match exists, else None.
    Never matches a segment to itself trivially — this is for *correcting* spelling,
    so an exact match just passes through unchanged (handled by caller)."""
    if not candidates:
        return None
    if _HAS_RAPIDFUZZ:
        result = process.extractOne(segment, candidates, scorer=fuzz.ratio)
        if result is None:
            return None
        match, score, _ = result
        score = score / 100.0
    else:
        matches = difflib.get_close_matches(segment, candidates, n=1, cutoff=threshold)
        if not matches:
            return None
        match = matches[0]
        score = difflib.SequenceMatcher(None, segment, match).ratio()

    if score >= threshold and match.lower() != segment.lower():
        return match, score
    return None


def correct_locality(loc: str, lexicon_terms: list[str], threshold: float) -> tuple[str, bool]:
    """Correct each segment of a predicted locality against the lexicon.
    Returns (corrected_string, was_changed)."""
    if not isinstance(loc, str) or loc.strip().upper() in ("MISSING", "NULL", ""):
        return loc, False

    changed = False
    out_cards = []
    for card in loc.split("|"):
        card_stripped = strip_substrate_phrases(card)
        parts = [p.strip(" .") for p in re.split(r"[,]", card_stripped) if p.strip(" .")]
        corrected_parts = []
        for part in parts:
            if part.lower() in STOPWORDS:
                corrected_parts.append(part)
                continue
            match = best_match(part, lexicon_terms, threshold)
            if match is not None:
                corrected_parts.append(match[0])
                changed = True
            else:
                corrected_parts.append(part)
        out_cards.append(", ".join(corrected_parts) if corrected_parts else card_stripped)
    return " | ".join(out_cards), changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--preds", required=True, help="predictions CSV with a verbatimLocality column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gazetteer", default=None, help="optional extra place-name list, one per line")
    ap.add_argument("--threshold", type=float, default=0.82,
                     help="min fuzzy similarity (0-1) required to apply a correction; "
                          "higher = more conservative")
    ap.add_argument("--min-freq", type=int, default=1,
                     help="minimum occurrences in train.csv for a term to be trusted "
                          "as a correction target (helps avoid overfitting to typos "
                          "in the 200-row training set itself)")
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    preds_df = pd.read_csv(args.preds)

    lexicon_counts = build_lexicon_from_train(train_df)
    lexicon_terms = [term for term, cnt in lexicon_counts.items() if cnt >= args.min_freq]

    if args.gazetteer:
        lexicon_terms = list(set(lexicon_terms) | load_external_gazetteer(args.gazetteer))

    print(f"Lexicon size: {len(lexicon_terms)} terms "
          f"({'rapidfuzz' if _HAS_RAPIDFUZZ else 'difflib fallback'})")

    corrected, flags = [], []
    for loc in preds_df["verbatimLocality"]:
        new_loc, was_changed = correct_locality(loc, lexicon_terms, args.threshold)
        corrected.append(new_loc)
        flags.append(was_changed)

    preds_df["verbatimLocality_original"] = preds_df["verbatimLocality"]
    preds_df["verbatimLocality"] = corrected
    preds_df["locality_lexicon_corrected"] = flags

    n_changed = sum(flags)
    print(f"Corrected {n_changed}/{len(preds_df)} locality predictions "
          f"({n_changed / max(len(preds_df),1):.1%})")
    if n_changed:
        sample = preds_df[preds_df["locality_lexicon_corrected"]][
            ["image_file", "verbatimLocality_original", "verbatimLocality"]
        ].head(10)
        print("Sample corrections:")
        print(sample.to_string(index=False))

    preds_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
