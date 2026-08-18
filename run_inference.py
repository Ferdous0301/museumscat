"""
Gemini VLM inference pipeline for the Danish Dung Beetle
Label Transcription Kaggle competition.
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


MODEL = "gemini-3.6-flash"


# ---------------------------------------------------------------------
# Gemini response schema
# ---------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verbatimDate": {
            "type": "STRING",
            "description": "Collection date exactly as written, or MISSING."
        },
        "verbatimLocality": {
            "type": "STRING",
            "description": "Collection locality exactly as written, or MISSING."
        },
        "date_confidence": {
            "type": "NUMBER",
            "description": "Confidence from 0.0 to 1.0."
        },
        "locality_confidence": {
            "type": "NUMBER",
            "description": "Confidence from 0.0 to 1.0."
        },
        "reasoning": {
            "type": "STRING",
            "description": "One short sentence describing any ambiguity."
        },
    },
    "required": [
        "verbatimDate",
        "verbatimLocality",
        "date_confidence",
        "locality_confidence",
        "reasoning",
    ],
}


# ---------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------

def encode_image(path: Path) -> tuple[bytes, str]:
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


# ---------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------

def build_fewshot_examples(
    train_df: pd.DataFrame,
    images_dir: Path,
    n: int,
    seed: int = 0,
) -> list:

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
        train_df["verbatimDate"]
        .fillna("")
        .astype(str)
        .eq("MISSING")
        |
        train_df["verbatimLocality"]
        .fillna("")
        .astype(str)
        .eq("MISSING")
    )

    pipe_rows = train_df[pipe_mask]
    missing_rows = train_df[missing_mask]

    picks = []

    if len(pipe_rows) > 0:
        picks.append(
            pipe_rows.sample(
                n=1,
                random_state=seed
            ).iloc[0]
        )

    if len(missing_rows) > 0:
        picks.append(
            missing_rows.sample(
                n=1,
                random_state=seed + 1
            ).iloc[0]
        )

    picked_indices = {row.name for row in picks}

    remaining = train_df[
        ~train_df.index.isin(picked_indices)
    ]

    remaining_needed = max(
        0,
        n - len(picks)
    )

    if remaining_needed > 0 and len(remaining) > 0:

        sample_n = min(
            remaining_needed,
            len(remaining)
        )

        sampled = remaining.sample(
            n=sample_n,
            random_state=seed + 2
        )

        picks.extend(
            [row for _, row in sampled.iterrows()]
        )

    parts = []

    for row in picks[:n]:

        image_file = str(row["image_file"])
        image_path = images_dir / image_file

        if not image_path.exists():
            print(
                f"WARNING: few-shot image not found: "
                f"{image_path}"
            )
            continue

        image_bytes, media_type = encode_image(
            image_path
        )

        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=media_type,
            )
        )

        example_answer = {
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

        parts.append(
            types.Part.from_text(
                text=(
                    "Correct transcription for this example:\n"
                    + json.dumps(
                        example_answer,
                        ensure_ascii=False
                    )
                )
            )
        )

    return parts


# ---------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------

def clean_result(result: dict) -> dict:

    date = result.get(
        "verbatimDate",
        "MISSING"
    )

    locality = result.get(
        "verbatimLocality",
        "MISSING"
    )

    if date is None or not str(date).strip():
        date = "MISSING"

    if locality is None or not str(locality).strip():
        locality = "MISSING"

    try:
        date_confidence = float(
            result.get(
                "date_confidence",
                0.5
            )
        )
    except (TypeError, ValueError):
        date_confidence = 0.5

    try:
        locality_confidence = float(
            result.get(
                "locality_confidence",
                0.5
            )
        )
    except (TypeError, ValueError):
        locality_confidence = 0.5

    date_confidence = max(
        0.0,
        min(1.0, date_confidence)
    )

    locality_confidence = max(
        0.0,
        min(1.0, locality_confidence)
    )

    return {
        "verbatimDate": str(date),
        "verbatimLocality": str(locality),
        "date_confidence": date_confidence,
        "locality_confidence": locality_confidence,
        "reasoning": str(
            result.get("reasoning", "")
        ),
    }


# ---------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------

def transcribe_one(
    client,
    image_path: Path,
    fewshot_parts: list,
    n_fewshot: int,
    retries: int = 3,
) -> dict:

    image_bytes, media_type = encode_image(
        image_path
    )

    parts = []

    if fewshot_parts:

        parts.append(
            types.Part.from_text(
                text=FEWSHOT_INSTRUCTION.format(
                    n=n_fewshot
                )
            )
        )

        parts.extend(fewshot_parts)

    parts.append(
        types.Part.from_text(
            text=USER_PROMPT_TEMPLATE
        )
    )

    parts.append(
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=media_type
        )
    )

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,

                    # Short structured answer; don't waste
                    # generation budget on unnecessary reasoning.
                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),

                    temperature=0.2,

                    max_output_tokens=1000,

                    response_mime_type="application/json",

                    response_schema=RESPONSE_SCHEMA,
                ),
            )

            # ---------------------------------------------------------
            # Prefer SDK-parsed response when available
            # ---------------------------------------------------------

            if getattr(response, "parsed", None) is not None:

                parsed = response.parsed

                if hasattr(parsed, "model_dump"):
                    parsed = parsed.model_dump()

                elif not isinstance(parsed, dict):
                    parsed = dict(parsed)

                return clean_result(parsed)

            # ---------------------------------------------------------
            # Fallback to response.text
            # ---------------------------------------------------------

            text = response.text

            if not text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            parsed = json.loads(text)

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True
    )

    parser.add_argument(
        "--images",
        required=True
    )

    parser.add_argument(
        "--out",
        required=True
    )

    parser.add_argument(
        "--train-csv",
        required=True
    )

    parser.add_argument(
        "--n-fewshot",
        type=int,
        default=4
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=1
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # API key
    # ---------------------------------------------------------------

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(
        api_key=api_key
    )

    # ---------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------

    images_dir = Path(args.images)

    df = pd.read_csv(args.csv)

    train_df = pd.read_csv(
        args.train_csv
    )

    if "image_file" not in df.columns:
        raise ValueError(
            "Input CSV must contain image_file."
        )

    required_train_columns = {
        "image_file",
        "verbatimDate",
        "verbatimLocality",
    }

    missing = (
        required_train_columns
        - set(train_df.columns)
    )

    if missing:
        raise ValueError(
            f"Training CSV missing columns: {missing}"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Input rows: {len(df)}"
    )

    print(
        f"Images directory: {images_dir}"
    )

    print(
        f"Few-shot examples: {args.n_fewshot}"
    )

    print(
        f"Runs per image: {args.n_runs}"
    )

    # ---------------------------------------------------------------
    # Few-shot
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    rows = []

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        image_file = str(
            row["image_file"]
        )

        image_path = (
            images_dir / image_file
        )

        if not image_path.exists():

            print(
                f"WARNING: image not found: "
                f"{image_path}"
            )

            result = {
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence": 0.0,
                "locality_confidence": 0.0,
                "reasoning": "Image not found.",
            }

        else:

            runs = []

            for _ in range(args.n_runs):

                result = transcribe_one(
                    client=client,
                    image_path=image_path,
                    fewshot_parts=fewshot_parts,
                    n_fewshot=args.n_fewshot,
                )

                runs.append(result)

            primary = runs[0]

            # -------------------------------------------------------
            # Self-consistency
            # -------------------------------------------------------

            if args.n_runs > 1:

                dates = {
                    r["verbatimDate"]
                    for r in runs
                }

                localities = {
                    r["verbatimLocality"]
                    for r in runs
                }

                if len(dates) > 1:

                    primary["date_confidence"] = min(
                        primary["date_confidence"],
                        0.4
                    )

                if len(localities) > 1:

                    primary["locality_confidence"] = min(
                        primary["locality_confidence"],
                        0.4
                    )

            result = primary

        rows.append(
            {
                "image_file": image_file,

                "verbatimDate":
                    result["verbatimDate"],

                "verbatimLocality":
                    result["verbatimLocality"],

                "date_confidence_raw":
                    result["date_confidence"],

                "locality_confidence_raw":
                    result["locality_confidence"],
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

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output = pd.DataFrame(rows)

    output.to_csv(
        args.out,
        index=False
    )

    print(
        f"\nWrote {len(output)} rows to:"
        f"\n{args.out}"
    )


if __name__ == "__main__":
    main()