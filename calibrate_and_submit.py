"""
Step 2: Calibrate raw self-reported VLM confidence against actual NED, then
produce a submission-ready CSV.

Why this matters: AURC punishes "confidently wrong" far more than "uncertain and
wrong". Raw self-reported confidence from a VLM is usually poorly calibrated
(e.g. clustered near 0.8-0.95 regardless of actual correctness). Isotonic
regression learns a monotonic mapping from (raw confidence) -> (empirical
1 - NED), fixing this cheaply using only your labeled 200 training examples.

Workflow:
1. Run run_inference.py on train.csv (images you already have ground truth for)
   to get raw confidences and predictions.
2. Compute actual NED per prediction against the ground truth.
3. Fit isotonic regression: raw_confidence -> (1 - NED)  [i.e. predicted "correctness"]
4. Run run_inference.py on test.csv to get raw confidences + predictions.
5. Apply the fitted calibrators to test raw confidences.
6. Write final submission.csv.

Usage:
    python calibrate_and_submit.py \
        --train-preds train_preds_raw.csv --train-truth train.csv \
        --test-preds test_preds_raw.csv --out submission.csv
"""
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


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
    """Normalized edit distance with pipe-ordering search (best of all orderings)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-preds", required=True, help="VLM predictions on train.csv images")
    ap.add_argument("--train-truth", required=True, help="original train.csv with ground truth")
    ap.add_argument("--test-preds", required=True, help="VLM predictions on test.csv images")
    ap.add_argument("--out", required=True, help="final submission CSV")
    args = ap.parse_args()

    train_preds = pd.read_csv(args.train_preds)
    train_truth = pd.read_csv(args.train_truth)
    test_preds = pd.read_csv(args.test_preds)

    merged = train_preds.merge(train_truth, on="image_file", suffixes=("_pred", "_true"))

    merged["date_ned"] = merged.apply(
        lambda r: ned(r["verbatimDate_pred"], r["verbatimDate_true"], is_date=True), axis=1)
    merged["loc_ned"] = merged.apply(
        lambda r: ned(r["verbatimLocality_pred"], r["verbatimLocality_true"], is_date=False), axis=1)

    print(f"Train mean NED: date={merged['date_ned'].mean():.4f} "
          f"locality={merged['loc_ned'].mean():.4f}")

    # Isotonic regression: raw confidence -> empirical correctness (1 - NED)
    date_iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    date_iso.fit(merged["date_confidence_raw"], 1 - merged["date_ned"])

    loc_iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    loc_iso.fit(merged["locality_confidence_raw"], 1 - merged["loc_ned"])

    # sanity check: calibrated confidence should correlate with actual correctness
    merged["date_conf_cal"] = date_iso.predict(merged["date_confidence_raw"])
    merged["loc_conf_cal"] = loc_iso.predict(merged["locality_confidence_raw"])
    print("Calibration check (train, should be monotonic-ish and near actual 1-NED):")
    print(merged[["date_confidence_raw", "date_conf_cal", "date_ned"]].sort_values("date_confidence_raw").to_string(index=False))

    # Apply to test predictions
    test_preds["date_confidence"] = date_iso.predict(test_preds["date_confidence_raw"])
    test_preds["locality_confidence"] = loc_iso.predict(test_preds["locality_confidence_raw"])

    # Guard against literal empty strings (Kaggle requires "MISSING")
    for col in ("verbatimDate", "verbatimLocality"):
        test_preds[col] = test_preds[col].fillna("MISSING").replace("", "MISSING")

    submission = test_preds[[
        "image_file", "verbatimDate", "date_confidence",
        "verbatimLocality", "locality_confidence",
    ]].rename(columns={
        "date_confidence": "verbatimDate_confidence",
        "locality_confidence": "verbatimLocality_confidence",
    })

    submission.to_csv(args.out, index=False)
    print(f"Wrote submission to {args.out}")


if __name__ == "__main__":
    main()
