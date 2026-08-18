"""
Gemini VLM-based transcription pipeline for the Danish dung beetle
specimen label competition.

Example:

python run_inference.py \
    --csv train.csv \
    --images images \
    --out train_preds_raw.csv \
    --train-csv train.csv \
    --n-fewshot 4 \
    --limit 10

For test inference:

python run_inference.py \
    --csv test.csv \
    --images images \
    --out test_preds_raw.csv \
    --train-csv train.csv \
    --n-fewshot 4

The script uses the labeled train.csv only for few-shot examples.
"""

import argparse
import base64
import json
import os
import random
import re
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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.6-flash"


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(path: Path) -> tuple[str, str]:
    """
    Read an image and convert it to base64 for Gemini.
    """

    ext = path.suffix.lower().lstrip(".")

    media_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }

    media_type = media_map.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, media_type


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text: str) -> dict:
    """
    Extract a JSON object from Gemini's response.

    Handles:
    - normal JSON
    - ```json ... ```
    - accidental surrounding text
    """

    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    # Remove markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # First attempt: entire response is JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: find the first complete JSON object.
    start = text.find("{")

    if start == -1:
        raise ValueError(
            f"Could not find JSON object in Gemini response:\n{text[:500]}"
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    raise ValueError(
        f"Could not parse JSON from Gemini response:\n{text[:500]}"
    )


# ============================================================
# VALIDATE RESULT
# ============================================================

def clean_result(result: dict) -> dict:
    """
    Make sure Gemini returned the fields expected by the competition.
    """

    date = result.get("verbatimDate", "MISSING")
    locality = result.get("verbatimLocality", "MISSING")

    date_conf = result.get("date_confidence", 0.0)
    locality_conf = result.get("locality_confidence", 0.0)

    if date is None or str(date).strip() == "":
        date = "MISSING"

    if locality is None or str(locality).strip() == "":
        locality = "MISSING"

    try:
        date_conf = float(date_conf)
    except Exception:
        date_conf = 0.0

    try:
        locality_conf = float(locality_conf)
    except Exception:
        locality_conf = 0.0

    date_conf = max(0.0, min(1.0, date_conf))
    locality_conf = max(0.0, min(1.0, locality_conf))

    return {
        "verbatimDate": str(date),
        "verbatimLocality": str(locality),
        "date_confidence": date_conf,
        "locality_confidence": locality_conf,
    }


# ============================================================
# FEW-SHOT SELECTION
# ============================================================

def build_fewshot_examples(
    train_df: pd.DataFrame,
    images_dir: Path,
    n: int,
    seed: int = 0,
):
    """
    Select useful training examples instead of purely random examples.

    Priority:
    1. multi-value examples
    2. MISSING examples
    3. examples containing short/abbreviated localities
    4. random remaining examples
    """

    rng = random.Random(seed)

    if n <= 0:
        return []

    df = train_df.copy()

    # Ensure columns exist.
    for col in ["verbatimDate", "verbatimLocality"]:
        if col not in df.columns:
            df[col] = "MISSING"

    df["verbatimDate"] = df["verbatimDate"].fillna("MISSING").astype(str)
    df["verbatimLocality"] = (
        df["verbatimLocality"].fillna("MISSING").astype(str)
    )

    selected_indices = []

    # --------------------------------------------------------
    # 1. Multi-card / multi-value examples
    # --------------------------------------------------------

    multi_mask = (
        df["verbatimDate"].str.contains(r"\|", regex=True, na=False)
        | df["verbatimLocality"].str.contains(r"\|", regex=True, na=False)
    )

    multi_rows = df[multi_mask]

    if len(multi_rows):
        idx = rng.choice(list(multi_rows.index))
        selected_indices.append(idx)

    # --------------------------------------------------------
    # 2. MISSING examples
    # --------------------------------------------------------

    missing_mask = (
        (df["verbatimDate"] == "MISSING")
        | (df["verbatimLocality"] == "MISSING")
    )

    missing_rows = df[missing_mask]

    if len(missing_rows):
        available = [
            idx for idx in missing_rows.index
            if idx not in selected_indices
        ]

        if available:
            selected_indices.append(rng.choice(available))

    # --------------------------------------------------------
    # 3. Short/abbreviated locality examples
    # --------------------------------------------------------

    short_loc_mask = df["verbatimLocality"].apply(
        lambda x: (
            x != "MISSING"
            and (
                len(x) <= 8
                or "." in x
                or "|" in x
            )
        )
    )

    short_rows = df[short_loc_mask]

    if len(short_rows):
        available = [
            idx for idx in short_rows.index
            if idx not in selected_indices
        ]

        if available:
            selected_indices.append(rng.choice(available))

    # --------------------------------------------------------
    # 4. Fill remaining slots randomly
    # --------------------------------------------------------

    remaining = [
        idx for idx in df.index
        if idx not in selected_indices
    ]

    rng.shuffle(remaining)

    for idx in remaining:
        if len(selected_indices) >= n:
            break

        selected_indices.append(idx)

    selected_indices = selected_indices[:n]

    # --------------------------------------------------------
    # Convert examples into Gemini content
    # --------------------------------------------------------

    blocks = []

    for idx in selected_indices:

        row = df.loc[idx]

        image_name = row.get("image_file")

        if not isinstance(image_name, str):
            continue

        img_path = images_dir / image_name

        if not img_path.exists():
            continue

        try:
            data, media_type = encode_image(img_path)
        except Exception as e:
            print(
                f"Warning: could not encode few-shot image "
                f"{image_name}: {e}"
            )
            continue

        blocks.append(
            types.Part.from_bytes(
                data=base64.b64decode(data),
                mime_type=media_type,
            )
        )

        example_json = {
            "verbatimDate": row["verbatimDate"],
            "verbatimLocality": row["verbatimLocality"],
        }

        blocks.append(
            types.Part.from_text(
                text=json.dumps(
                    example_json,
                    ensure_ascii=False,
                )
            )
        )

    return blocks


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def transcribe_one(
    client,
    image_path: Path,
    fewshot_blocks,
    n_fewshot: int,
    retries: int = 3,
):
    """
    Send one specimen image to Gemini.

    Important:
    429 quota errors are NOT retried repeatedly because doing so
    wastes time and cannot recover an exhausted daily quota.
    """

    data, media_type = encode_image(image_path)

    content = []

    if fewshot_blocks:
        content.append(
            types.Part.from_text(
                text=FEWSHOT_INSTRUCTION.format(n=n_fewshot)
            )
        )

        content.extend(fewshot_blocks)

    content.append(
        types.Part.from_text(
            text=USER_PROMPT_TEMPLATE
        )
    )

    content.append(
        types.Part.from_bytes(
            data=base64.b64decode(data),
            mime_type=media_type,
        )
    )

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                    max_output_tokens=300,
                ),
            )

            text = response.text or ""

            parsed = extract_json(text)

            return clean_result(parsed)

        except Exception as e:

            last_error = e

            error_text = str(e)

            # ------------------------------------------------
            # Quota error
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "quota" in error_text.lower()
            ):
                print(
                    f"Quota exhausted while processing "
                    f"{image_path.name}."
                )

                return {
                    "verbatimDate": "MISSING",
                    "verbatimLocality": "MISSING",
                    "date_confidence": 0.0,
                    "locality_confidence": 0.0,
                    "_quota_error": True,
                }

            # ------------------------------------------------
            # Other errors
            # ------------------------------------------------

            print(
                f"Attempt {attempt}/{retries} failed for "
                f"{image_path.name}: {error_text[:300]}"
            )

            if attempt < retries:
                time.sleep(2 ** (attempt - 1))

    return {
        "verbatimDate": "MISSING",
        "verbatimLocality": "MISSING",
        "date_confidence": 0.0,
        "locality_confidence": 0.0,
        "_error": str(last_error),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing image_file column",
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing specimen images",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--train-csv",
        required=True,
        help="train.csv used for few-shot examples",
    )

    parser.add_argument(
        "--n-fewshot",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of independent calls per image",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # ========================================================
    # Gemini client
    # ========================================================

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the environment. "
            "Set it as a Kaggle secret named GEMINI_API_KEY."
        )

    client = genai.Client(api_key=api_key)

    # ========================================================
    # Paths
    # ========================================================

    images_dir = Path(args.images)

    df = pd.read_csv(args.csv)
    train_df = pd.read_csv(args.train_csv)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Input rows: {len(df)}")
    print(f"Images directory: {images_dir}")
    print(f"Few-shot examples: {args.n_fewshot}")
    print(f"Runs per image: {args.n_runs}")
    print("=" * 70)

    # ========================================================
    # Build few-shot context
    # ========================================================

    fewshot_blocks = build_fewshot_examples(
        train_df=train_df,
        images_dir=images_dir,
        n=args.n_fewshot,
        seed=args.seed,
    )

    print(
        f"Built few-shot context with approximately "
        f"{args.n_fewshot} examples."
    )

    # ========================================================
    # Inference
    # ========================================================

    rows = []

    quota_exhausted = False

    for position, (_, row) in enumerate(df.iterrows(), start=1):

        image_file = row["image_file"]

        image_path = images_dir / image_file

        if not image_path.exists():

            print(
                f"[{position}/{len(df)}] {image_file} "
                f"-> IMAGE NOT FOUND"
            )

            rows.append({
                "image_file": image_file,
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence_raw": 0.0,
                "locality_confidence_raw": 0.0,
            })

            continue

        runs = []

        for run_index in range(args.n_runs):

            result = transcribe_one(
                client=client,
                image_path=image_path,
                fewshot_blocks=fewshot_blocks,
                n_fewshot=args.n_fewshot,
            )

            runs.append(result)

            if result.get("_quota_error"):
                quota_exhausted = True
                break

        # ----------------------------------------------------
        # If quota is exhausted, don't send additional requests.
        # ----------------------------------------------------

        if quota_exhausted:

            primary = {
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence": 0.0,
                "locality_confidence": 0.0,
            }

            print(
                f"[{position}/{len(df)}] {image_file} "
                f"-> QUOTA EXHAUSTED"
            )

        else:

            primary = runs[0]

            # ------------------------------------------------
            # Self-consistency when n_runs > 1
            # ------------------------------------------------

            if args.n_runs > 1:

                dates = {
                    r.get("verbatimDate", "MISSING")
                    for r in runs
                }

                localities = {
                    r.get("verbatimLocality", "MISSING")
                    for r in runs
                }

                if len(dates) > 1:

                    primary["date_confidence"] = min(
                        primary.get("date_confidence", 0.5),
                        0.4,
                    )

                if len(localities) > 1:

                    primary["locality_confidence"] = min(
                        primary.get("locality_confidence", 0.5),
                        0.4,
                    )

            print(
                f"[{position}/{len(df)}] {image_file} "
                f"-> "
                f"date={primary.get('verbatimDate', 'MISSING')!r} "
                f"loc={primary.get('verbatimLocality', 'MISSING')!r} "
                f"date_conf={primary.get('date_confidence', 0.0):.2f} "
                f"loc_conf={primary.get('locality_confidence', 0.0):.2f}"
            )

        rows.append({
            "image_file": image_file,
            "verbatimDate": primary.get(
                "verbatimDate",
                "MISSING",
            ),
            "verbatimLocality": primary.get(
                "verbatimLocality",
                "MISSING",
            ),
            "date_confidence_raw": primary.get(
                "date_confidence",
                0.0,
            ),
            "locality_confidence_raw": primary.get(
                "locality_confidence",
                0.0,
            ),
        })

        # ----------------------------------------------------
        # Stop immediately once quota is exhausted.
        # ----------------------------------------------------

        if quota_exhausted:
            print()
            print("=" * 70)
            print(
                "Gemini quota is exhausted. "
                "Stopping inference instead of generating "
                "more failed requests."
            )
            print("=" * 70)
            break

    # ========================================================
    # Save output
    # ========================================================

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        args.out,
        index=False,
    )

    print()
    print(
        f"Wrote {len(out_df)} rows to:"
    )
    print(args.out)

    if len(out_df) < len(df):
        print(
            f"WARNING: only {len(out_df)} / {len(df)} rows "
            f"were processed because the Gemini quota was exhausted."
        )


if __name__ == "__main__":
    main()