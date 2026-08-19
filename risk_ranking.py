"""
Step 2 (replacement for the isotonic-only approach in calibrate_and_submit.py):
combine multiple independent signals into one risk score per field, then use
that score -- not raw VLM self-reported confidence -- to rank predictions by
trustworthiness for AURC.

WHY THIS EXISTS (vs. calibrate_and_submit.py alone):
calibrate_and_submit.py fixes *miscalibration* of a single signal (raw VLM
confidence) via isotonic regression: it learns that "when the model says
0.9, it's actually right ~70% of the time" and remaps accordingly. That
helps, but it can't fix a confidently-wrong case where the ONE signal it has
(self-reported confidence) is simply wrong -- e.g. the VLM said 0.90 for a
date it misread as "22.S.1977." instead of "22.5.1977.". No amount of
remapping a single bad number produces a better estimate.

This script instead builds a small feature vector per field from several
INDEPENDENT signals, each of which can catch errors the others miss:
    1. raw_confidence       -- the VLM's own self-report
    2. verify_changed       -- did the --verify second pass change the
                                answer? A field that survived a second,
                                differently-worded look unchanged is more
                                trustworthy than one whose first attempt
                                already needed correcting.
    3. is_missing            -- MISSING fields already get clamped to a very
                                low score elsewhere, but including it as a
                                feature lets the model learn its own weight
                                rather than a hardcoded 0.05.
    4. value_length           -- pathologically short non-MISSING answers
                                (e.g. a single character) in the date field
                                are almost always wrong.
    5. lexicon_edit_distance (locality only) -- distance to the nearest
                                lexicon term from locality_lexicon.py.
                                Reuses that file's tokenizer/matcher rather
                                than reimplementing it.
    6. ocr_agreement (optional) -- if ocr_crosscheck.py has been run and its
                                output columns are present, how closely the
                                VLM's answer agrees with an independent OCR
                                reading of the same pixels. This is the
                                single most powerful signal for catching
                                character-level misreads like the S/5 case,
                                since it comes from a genuinely different
                                model architecture rather than the same VLM
                                double-checking itself.

A separate small logistic regression is fit per field (date, locality) on
your labeled train rows, predicting P(correct) from these features, using
actual NED against ground truth as the training label. That's then applied
to test predictions to produce a calibrated, multi-signal risk score.

If you haven't run --verify or ocr_crosscheck.py, those columns simply won't
exist and their features are silently dropped -- the script degrades
gracefully to using whatever signals are actually available, down to just
raw confidence (equivalent to calibrate_and_submit.py) in the minimal case.

Usage:
    python risk_ranking.py \
        --train-preds train_preds_raw.csv --train-truth train.csv \
        --test-preds test_preds_raw.csv --out submission.csv \
        --train-csv train.csv   # for the locality lexicon
        [--test-ocr test_preds_ocr.csv]   # optional, from ocr_crosscheck.py
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from locality_lexicon import (
    build_lexicon_from_train,
    tokenize_locality,
)


# ================================================================
# NED (same definition as calibrate_and_submit.py, kept identical so
# risk_ranking.py and calibrate_and_submit.py agree on what "correct" means)
# ================================================================

def normalize_date_punct(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r"[.,\-·\s]+", " ", s).strip()


def edit_distance(a: str, b: str) -> int:
    a, b = a or "", b or ""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def ned(pred: str, truth: str, is_date: bool) -> float:
    pred = "" if pred == "MISSING" or pred is None else str(pred)
    truth = "" if truth == "MISSING" or truth is None else str(truth)

    pred_parts = [p.strip() for p in pred.split("|")] if pred else [""]
    truth_parts = [t.strip() for t in truth.split("|")] if truth else [""]

    if is_date:
        pred_parts = [normalize_date_punct(p) for p in pred_parts]
        truth_parts = [normalize_date_punct(t) for t in truth_parts]

    pred_joined = " ".join(sorted(p.lower() for p in pred_parts))
    truth_joined = " ".join(sorted(t.lower() for t in truth_parts))

    max_len = max(len(pred_joined), len(truth_joined), 1)
    return edit_distance(pred_joined, truth_joined) / max_len


# ================================================================
# FEATURE BUILDING
# ================================================================

try:
    from rapidfuzz import fuzz as _rf_fuzz, process as _rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib


def nearest_lexicon_distance(value: str, lexicon_terms: list) -> float:
    """
    Normalized edit distance (0 = exact match, 1 = totally different) from
    the predicted locality to the closest term in the training lexicon.
    Returns 0.5 (a neutral "unknown" value) for MISSING, since there's
    nothing to compare.

    Deliberately does NOT reuse locality_lexicon.best_match() here: that
    function is built for *correction* and intentionally returns None for
    an exact match (nothing to correct), which would make every exact
    match look maximally distant if used for scoring. This does its own
    direct fuzzy scoring instead.
    """
    if not isinstance(value, str) or value.strip().upper() in ("MISSING", ""):
        return 0.5

    segments = tokenize_locality(value)
    if not segments or not lexicon_terms:
        return 0.5

    best_dist = 1.0
    for seg in segments:
        if _HAS_RAPIDFUZZ:
            result = _rf_process.extractOne(seg, lexicon_terms, scorer=_rf_fuzz.ratio)
            score = (result[1] / 100.0) if result else 0.0
        else:
            matches = difflib.get_close_matches(seg, lexicon_terms, n=1, cutoff=0.0)
            score = difflib.SequenceMatcher(None, seg, matches[0]).ratio() if matches else 0.0
        best_dist = min(best_dist, 1.0 - score)

    return best_dist


def build_features(df: pd.DataFrame, field: str, lexicon_terms: list,
                    has_ocr: bool) -> pd.DataFrame:
    """
    Build the feature matrix for one field ('date' or 'locality').
    Degrades gracefully: any column that isn't present in df is simply
    left out of the feature set rather than raising.
    """
    value_col = "verbatimDate" if field == "date" else "verbatimLocality"
    conf_col = "date_confidence_raw" if field == "date" else "locality_confidence_raw"
    changed_col = "date_verify_changed" if field == "date" else "locality_verify_changed"
    ocr_agree_col = "date_ocr_agreement" if field == "date" else "locality_ocr_agreement"

    feats = pd.DataFrame(index=df.index)
    feats["raw_confidence"] = df[conf_col].astype(float)
    feats["is_missing"] = (df[value_col].astype(str).str.strip().str.upper() == "MISSING").astype(float)
    feats["value_length"] = df[value_col].astype(str).apply(lambda s: 0 if s.upper() == "MISSING" else len(s))

    if changed_col in df.columns:
        feats["verify_changed"] = df[changed_col].astype(bool).astype(float)
    else:
        feats["verify_changed"] = 0.0  # neutral default when --verify wasn't used

    if field == "locality":
        feats["lexicon_distance"] = df[value_col].apply(
            lambda v: nearest_lexicon_distance(v, lexicon_terms)
        )

    if has_ocr and ocr_agree_col in df.columns:
        feats["ocr_agreement"] = df[ocr_agree_col].astype(float)

    return feats


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-preds", required=True, help="VLM predictions on train.csv images")
    ap.add_argument("--train-truth", required=True, help="original train.csv with ground truth")
    ap.add_argument("--test-preds", required=True, help="VLM predictions on test.csv images")
    ap.add_argument("--train-csv", required=True, help="train.csv, used to build the locality lexicon")
    ap.add_argument("--out", required=True, help="final submission CSV")
    ap.add_argument("--train-ocr", default=None,
                     help="optional ocr_crosscheck.py output for the train predictions "
                          "(adds the ocr_agreement feature if given)")
    ap.add_argument("--test-ocr", default=None,
                     help="optional ocr_crosscheck.py output for the test predictions")
    args = ap.parse_args()

    train_preds = pd.read_csv(args.train_preds)
    train_truth = pd.read_csv(args.train_truth)
    test_preds = pd.read_csv(args.test_preds)
    lexicon_source_df = pd.read_csv(args.train_csv)

    if args.train_ocr:
        train_ocr = pd.read_csv(args.train_ocr)
        train_preds = train_preds.merge(train_ocr, on="image_file", how="left", suffixes=("", "_ocrdup"))
    if args.test_ocr:
        test_ocr = pd.read_csv(args.test_ocr)
        test_preds = test_preds.merge(test_ocr, on="image_file", how="left", suffixes=("", "_ocrdup"))

    has_ocr = args.train_ocr is not None and args.test_ocr is not None
    if bool(args.train_ocr) != bool(args.test_ocr):
        print("WARNING: only one of --train-ocr/--test-ocr given; "
              "OCR agreement feature will not be used (need both to fit and apply it consistently).")

    lexicon_counts = build_lexicon_from_train(lexicon_source_df)
    lexicon_terms = [term for term, cnt in lexicon_counts.items() if cnt >= 1]
    print(f"Lexicon size for risk features: {len(lexicon_terms)} terms")

    merged = train_preds.merge(train_truth, on="image_file", suffixes=("_pred", "_true"))

    merged["date_ned"] = merged.apply(
        lambda r: ned(r["verbatimDate_pred"], r["verbatimDate_true"], is_date=True), axis=1)
    merged["loc_ned"] = merged.apply(
        lambda r: ned(r["verbatimLocality_pred"], r["verbatimLocality_true"], is_date=False), axis=1)

    # Binary "correct" label for classification: NED below a small threshold
    # counts as correct. A hard 0/1 label is simpler and more robust than
    # regressing directly on (1-NED) when you only have ~200 rows.
    CORRECT_NED_THRESHOLD = 0.15
    merged["date_correct"] = (merged["date_ned"] <= CORRECT_NED_THRESHOLD).astype(int)
    merged["loc_correct"] = (merged["loc_ned"] <= CORRECT_NED_THRESHOLD).astype(int)

    print(f"Train accuracy @ NED<={CORRECT_NED_THRESHOLD}: "
          f"date={merged['date_correct'].mean():.3f} locality={merged['loc_correct'].mean():.3f}")

    # Reconstruct field-named columns for feature building (merge added
    # _pred/_true suffixes to the shared column names).
    merged_for_feats = merged.rename(columns={
        "verbatimDate_pred": "verbatimDate",
        "verbatimLocality_pred": "verbatimLocality",
    })

    models = {}
    safety_iso = {}  # always fit, used to blend/dampen the logistic model's extrapolation

    # How much weight the logistic risk model gets vs. the safety floor
    # (plain isotonic calibration of raw_confidence alone). 0.6 means the
    # richer multi-signal model dominates when it agrees with the simple
    # baseline, but a wild disagreement gets pulled back toward the safer
    # single-signal estimate rather than trusted outright -- found to be
    # necessary during testing: an unregularized-ish fit can extrapolate
    # to near-0 or near-1 confidence on inputs unlike anything seen in
    # training (e.g. a small feature set separating training data almost
    # perfectly on an incidental feature like value_length), which is
    # exactly the kind of overconfident-and-wrong behavior AURC punishes
    # most.
    LOGISTIC_BLEND_WEIGHT = 0.6

    for field, label_col in (("date", "date_correct"), ("locality", "loc_correct")):
        X = build_features(merged_for_feats, field, lexicon_terms, has_ocr)
        y = merged_for_feats[label_col]

        # Always fit the safety floor, regardless of whether the richer
        # model is usable.
        safety = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        safety.fit(X["raw_confidence"], y)
        safety_iso[field] = safety

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos < 5 or n_neg < 5:
            # Too few examples of one class for a stable logistic fit on a
            # ~200-row train set -- use the safety floor alone.
            print(f"[{field}] Not enough class balance for logistic risk model "
                  f"(pos={n_pos}, neg={n_neg}); using isotonic-on-confidence only.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0.0))

        # C=0.3 (stronger-than-default L2 regularization): with only a
        # handful of hand-built features on ~200 rows, one of them can
        # easily separate the training data almost perfectly by accident,
        # which drives unregularized logistic coefficients toward
        # infinity. Stronger regularization keeps predicted probabilities
        # from saturating near 0/1 on unfamiliar test-time inputs.
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)
        clf.fit(X_scaled, y)
        models[field] = (clf, scaler, list(X.columns))

        train_pred_proba = clf.predict_proba(X_scaled)[:, 1]
        print(f"[{field}] Logistic risk model fit on features: {list(X.columns)}")
        print(f"[{field}] Train-set correlation between predicted P(correct) and actual: "
              f"{np.corrcoef(train_pred_proba, y)[0, 1]:.3f}")
        print(f"[{field}] Predicted-probability range on train set: "
              f"[{train_pred_proba.min():.3f}, {train_pred_proba.max():.3f}] "
              f"(if this hugs [0,1], the model may still be overconfident -- "
              f"inspect before trusting on the full test set)")

    test_for_feats = test_preds.copy()

    for field, conf_out_col in (("date", "verbatimDate_confidence"), ("locality", "verbatimLocality_confidence")):
        Xt = build_features(test_for_feats, field, lexicon_terms, has_ocr)
        safety_conf = safety_iso[field].predict(Xt["raw_confidence"])

        if field in models:
            clf, scaler, feature_names = models[field]
            # Align columns exactly to what the model was fit on -- if OCR
            # features are missing at test time but were present at train
            # time (or vice versa), fill with a neutral value rather than
            # erroring, so this degrades gracefully instead of crashing.
            for col in feature_names:
                if col not in Xt.columns:
                    Xt[col] = 0.5 if col == "ocr_agreement" else 0.0
            Xt_ordered = Xt[feature_names].fillna(0.0)
            Xt_scaled = scaler.transform(Xt_ordered)
            logistic_conf = clf.predict_proba(Xt_scaled)[:, 1]

            risk_conf = (
                LOGISTIC_BLEND_WEIGHT * logistic_conf
                + (1 - LOGISTIC_BLEND_WEIGHT) * safety_conf
            )
        else:
            risk_conf = safety_conf

        test_preds[conf_out_col] = risk_conf

    # MISSING fields still get a hard floor regardless of what the risk
    # model predicted -- a learned model with limited training data
    # shouldn't be trusted to independently rediscover this constraint,
    # and it's a correctness requirement, not just a preference.
    date_missing_mask = test_preds["verbatimDate"].astype(str).str.strip().str.upper() == "MISSING"
    loc_missing_mask = test_preds["verbatimLocality"].astype(str).str.strip().str.upper() == "MISSING"
    test_preds.loc[date_missing_mask, "verbatimDate_confidence"] = \
        test_preds.loc[date_missing_mask, "verbatimDate_confidence"].clip(upper=0.05)
    test_preds.loc[loc_missing_mask, "verbatimLocality_confidence"] = \
        test_preds.loc[loc_missing_mask, "verbatimLocality_confidence"].clip(upper=0.05)

    for col in ("verbatimDate", "verbatimLocality"):
        test_preds[col] = test_preds[col].fillna("MISSING").replace("", "MISSING")

    submission = test_preds[[
        "image_file", "verbatimDate", "verbatimDate_confidence",
        "verbatimLocality", "verbatimLocality_confidence",
    ]]

    submission.to_csv(args.out, index=False)
    print(f"Wrote submission to {args.out}")


if __name__ == "__main__":
    main()