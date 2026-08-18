"""
Gemini VLM-based transcription pipeline for the Danish Dung Beetle
Label Transcription Kaggle competition.

Usage:
    python run_inference.py \
        --csv /path/to/test.csv \
        --images /path/to/images \
        --train-csv /path/to/train.csv \
        --n-fewshot 4 \
        --out /path/to/test_preds_raw.csv

Requires:
    pip install -U google-genai pandas

Kaggle:
    GEMINI_API_KEY must be configured as a Kaggle Secret.

This script uses the provided training labels only for few-shot prompting.
It does not manually annotate test data.
"""

import argparse
import base64
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from prompt_template import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FEWSHOT_INSTRUCTION,
)


# Gemini model available to the user's current API account.
MODEL = "gemini-3.6-flash"


def encode_image(path: Path) -> tuple[bytes, str]:
    """
    Read an image from disk and return raw bytes plus MIME type.
    """
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    elif ext == ".bmp":
        media_type = "image/bmp"
    else:
        raise ValueError(f"Unsupported image format: {ext}")

    with open(path, "rb") as f:
        data = f.read()

    return data, media_type


def build_fewshot_examples(
    train_df: pd.DataFrame,
    images_dir: Path,
    n: int,
    seed: int = 0,
) -> list:
    """
    Select a small set of diverse training examples.

    Preference is given to:
    - multi-card examples
    - MISSING examples
    - random examples

    Returns Gemini-compatible content parts.
    """

    rng = random.Random(seed)

    pipe_mask = (
        train_df["verbatimDate"]
        .fillna("")
        .astype(str)
        .str.contains(r"\|", regex=True)
        |
        train_df["verbatimLocality"]
        .fillna("")
        .astype(str)
        .str.contains(r"\|", regex=True)
    )

    missing_mask = (
        train_df["verbatimDate"].fillna("").astype(str).eq("MISSING")
        |
        train_df["verbatimLocality"].fillna("").astype(str).eq("MISSING")
    )

    pipe_rows = train_df[pipe_mask]
    missing_rows = train_df[missing_mask]

    picks = []

    if len(pipe_rows) > 0:
        picks.append(
            pipe_rows.sample(
                n=1,
                random_state=seed,
            ).iloc[0]
        )

    if len(missing_rows) > 0:
        picks.append(
            missing_rows.sample(
                n=1,
                random_state=seed + 1,
            ).iloc[0]
        )

    picked_indices = {row.name for row in picks}

    remaining_pool = train_df[
        ~train_df.index.isin(picked_indices)
    ]

    remaining_needed = max(0, n - len(picks))

    if remaining_needed > 0 and len(remaining_pool) > 0:
        sample_n = min(remaining_needed, len(remaining_pool))

        random_rows = remaining_pool.sample(
            n=sample_n,
            random_state=seed + 2,
        )

        picks.extend(
            [row for _, row in random_rows.iterrows()]
        )

    examples = []

    for row in picks[:n]:
        image_file = str(row["image_file"])
        image_path = images_dir / image_file

        if not image_path.exists():
            print(
                f"WARNING: Few-shot image not found: {image_path}"
            )
            continue

        image_bytes, media_type = encode_image(image_path)

        examples.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=media_type,
            )
        )

        answer = {
            "verbatimDate": (
                "MISSING"
                if pd.isna(row["verbatimDate"])
                else str(row["verbatimDate"])
            ),
            "verbatimLocality": (
                "MISSING"
                if pd.isna(row["verbatimLocality"])
                else str(row["verbatimLocality"])
            ),
        }

        examples.append(
            types.Part.from_text(
                text=json.dumps(
                    answer,
                    ensure_ascii=False,
                )
            )
        )

    return examples


def extract_json(text: str) -> dict:
    """
    Extract a JSON object from Gemini's response.

    Handles:
    - plain JSON
    - ```json ... ```
    - accidental surrounding text
    """

    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    # Remove markdown fences if Gemini adds them.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # First attempt: direct JSON parsing.
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Second attempt: locate the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]

        result = json.loads(candidate)

        if isinstance(result, dict):
            return result

    raise ValueError(
        f"Could not parse JSON from Gemini response:\n{text}"
    )


def clean_result(result: dict) -> dict:
    """
    Normalize Gemini's returned dictionary to the expected schema.
    """

    date = result.get("verbatimDate", "MISSING")
    locality = result.get("verbatimLocality", "MISSING")

    if date is None or str(date).strip() == "":
        date = "MISSING"

    if locality is None or str(locality).strip() == "":
        locality = "MISSING"

    try:
        date_confidence = float(
            result.get("date_confidence", 0.5)
        )
    except (TypeError, ValueError):
        date_confidence = 0.5

    try:
        locality_confidence = float(
            result.get("locality_confidence", 0.5)
        )
    except (TypeError, ValueError):
        locality_confidence = 0.5

    date_confidence = max(
        0.0,
        min(1.0, date_confidence),
    )

    locality_confidence = max(
        0.0,
        min(1.0, locality_confidence),
    )

    reasoning = str(
        result.get("reasoning", "")
    )

    return {
        "verbatimDate": str(date),
        "verbatimLocality": str(locality),
        "date_confidence": date_confidence,
        "locality_confidence": locality_confidence,
        "reasoning": reasoning,
    }


def transcribe_one(
    client: genai.Client,
    image_path: Path,
    fewshot_parts: list,
    n_fewshot: int,
    retries: int = 3,
) -> dict:
    """
    Send one specimen image to Gemini.
    """

    image_bytes, media_type = encode_image(image_path)

    content = []

    if fewshot_parts:
        content.append(
            types.Part.from_text(
                text=FEWSHOT_INSTRUCTION.format(
                    n=n_fewshot
                )
            )
        )

        content.extend(fewshot_parts)

    content.append(
        types.Part.from_text(
            text=USER_PROMPT_TEMPLATE
        )
    )

    content.append(
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=media_type,
        )
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=content,
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json",
                ),
            )

            text = response.text.strip()

            parsed = extract_json(text)

            return clean_result(parsed)

        except Exception as exc:
            print(
                f"Attempt {attempt + 1}/{retries} failed "
                f"for {image_path.name}: {exc}"
            )

            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return {
        "verbatimDate": "MISSING",
        "verbatimLocality": "MISSING",
        "date_confidence": 0.0,
        "locality_confidence": 0.0,
        "reasoning": "Gemini inference failed after retries.",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing image_file.",
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing specimen images.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--train-csv",
        required=True,
        help="Training CSV containing ground-truth labels.",
    )

    parser.add_argument(
        "--n-fewshot",
        type=int,
        default=4,
        help="Number of few-shot examples.",
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of Gemini calls per image.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows for debugging.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Gemini API
    # ------------------------------------------------------------------

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable was not found. "
            "Make sure your Kaggle Secret is enabled."
        )

    client = genai.Client(
        api_key=api_key
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    images_dir = Path(args.images)

    df = pd.read_csv(args.csv)
    train_df = pd.read_csv(args.train_csv)

    required_test_column = "image_file"

    if required_test_column not in df.columns:
        raise ValueError(
            f"Input CSV must contain '{required_test_column}'. "
            f"Found: {df.columns.tolist()}"
        )

    required_train_columns = {
        "image_file",
        "verbatimDate",
        "verbatimLocality",
    }

    missing_columns = (
        required_train_columns - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    print(f"Model: {MODEL}")
    print(f"Input rows: {len(df)}")
    print(f"Images directory: {images_dir}")
    print(f"Few-shot examples: {args.n_fewshot}")
    print(f"Runs per image: {args.n_runs}")

    # ------------------------------------------------------------------
    # Build few-shot context
    # ------------------------------------------------------------------

    fewshot_parts = build_fewshot_examples(
        train_df=train_df,
        images_dir=images_dir,
        n=args.n_fewshot,
        seed=0,
    )

    print(
        f"Built few-shot context with "
        f"{len(fewshot_parts) // 2} examples."
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    rows = []

    for position, (_, row) in enumerate(df.iterrows(), start=1):

        image_file = str(row["image_file"])
        image_path = images_dir / image_file

        if not image_path.exists():
            print(
                f"WARNING: Image not found: {image_path}"
            )

            result = {
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence": 0.0,
                "locality_confidence": 0.0,
                "reasoning": "Image file not found.",
            }

        else:
            runs = []

            for run_index in range(args.n_runs):
                result = transcribe_one(
                    client=client,
                    image_path=image_path,
                    fewshot_parts=fewshot_parts,
                    n_fewshot=args.n_fewshot,
                )

                runs.append(result)

            # ----------------------------------------------------------
            # Self-consistency signal
            # ----------------------------------------------------------

            primary = runs[0]

            if args.n_runs > 1:

                dates = {
                    r.get("verbatimDate", "")
                    for r in runs
                }

                localities = {
                    r.get("verbatimLocality", "")
                    for r in runs
                }

                if len(dates) > 1:
                    primary["date_confidence"] = min(
                        primary["date_confidence"],
                        0.4,
                    )

                if len(localities) > 1:
                    primary["locality_confidence"] = min(
                        primary["locality_confidence"],
                        0.4,
                    )

            result = primary

        rows.append(
            {
                "image_file": image_file,
                "verbatimDate": result["verbatimDate"],
                "verbatimLocality": result["verbatimLocality"],
                "date_confidence_raw": result[
                    "date_confidence"
                ],
                "locality_confidence_raw": result[
                    "locality_confidence"
                ],
            }
        )

        print(
            f"[{position}/{len(df)}] "
            f"{image_file} -> "
            f"date={result['verbatimDate']!r} "
            f"loc={result['verbatimLocality']!r} "
            f"date_conf={result['date_confidence']:.2f} "
            f"loc_conf={result['locality_confidence']:.2f}"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        args.out,
        index=False,
    )

    print()
    print(
        f"Wrote {len(out_df)} rows to:"
        f"\n{args.out}"
    )


if __name__ == "__main__":
    main()