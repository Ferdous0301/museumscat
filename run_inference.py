"""
Local VLM inference pipeline for the Danish Dung Beetle label
transcription task.

Uses:
    Qwen/Qwen2.5-VL-3B-Instruct

Runs locally on the Kaggle GPU.
No API key or external inference API is required.

Example:

python run_inference.py \
    --csv /kaggle/input/competitions/museumscat-specimen-collection-annotation-task/train.csv \
    --images /kaggle/input/competitions/museumscat-specimen-collection-annotation-task/images \
    --out train_preds_raw.csv \
    --train-csv /kaggle/input/competitions/museumscat-specimen-collection-annotation-task/train.csv \
    --n-fewshot 4
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from prompt_template import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FEWSHOT_INSTRUCTION,
)


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ---------------------------------------------------------------------
# IMAGE
# ---------------------------------------------------------------------

def check_image(path: Path):
    """Check that the image exists and can be opened."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise RuntimeError(f"Could not read image {path}: {e}")


# ---------------------------------------------------------------------
# FEW SHOT
# ---------------------------------------------------------------------

def build_fewshot_examples(
    train_df: pd.DataFrame,
    images_dir: Path,
    n: int,
    seed: int = 0,
):
    """
    Select a small number of useful training examples.

    We deliberately prefer:
      - multi-card examples
      - MISSING examples
      - ordinary examples

    Ground-truth answers are included as calibration examples.
    """

    rng = random.Random(seed)

    train_df = train_df.copy()

    # Multi-card examples
    pipe_mask = (
        train_df["verbatimDate"].astype(str).str.contains(r"\|", na=False)
        |
        train_df["verbatimLocality"].astype(str).str.contains(r"\|", na=False)
    )

    pipe_rows = train_df[pipe_mask]

    # Missing-value examples
    missing_mask = (
        (train_df["verbatimDate"].astype(str) == "MISSING")
        |
        (train_df["verbatimLocality"].astype(str) == "MISSING")
    )

    missing_rows = train_df[missing_mask]

    selected = []

    if len(pipe_rows) > 0:
        selected.append(
            pipe_rows.sample(
                1,
                random_state=seed
            ).iloc[0]
        )

    if len(missing_rows) > 0:
        candidate = missing_rows.sample(
            1,
            random_state=seed + 1
        ).iloc[0]

        if candidate["image_file"] not in {
            x["image_file"] for x in selected
        }:
            selected.append(candidate)

    # Fill remaining slots randomly
    used = {x["image_file"] for x in selected}

    remaining = train_df[
        ~train_df["image_file"].isin(used)
    ]

    while len(selected) < n and len(remaining) > 0:

        row = remaining.sample(
            1,
            random_state=seed + len(selected) + 10
        ).iloc[0]

        selected.append(row)

        remaining = remaining[
            remaining["image_file"] != row["image_file"]
        ]

    return selected[:n]


# ---------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------

def build_messages(
    image_path: Path,
    fewshot_examples,
    images_dir: Path,
):
    """
    Build the multimodal conversation for Qwen.
    """

    content = []

    # Strong task instructions
    content.append({
        "type": "text",
        "text": SYSTEM_PROMPT,
    })

    # Few-shot examples
    if fewshot_examples:

        content.append({
            "type": "text",
            "text": FEWSHOT_INSTRUCTION.format(
                n=len(fewshot_examples)
            ),
        })

        for row in fewshot_examples:

            example_path = images_dir / row["image_file"]

            if not example_path.exists():
                continue

            content.append({
                "type": "image",
                "image": str(example_path),
            })

            content.append({
                "type": "text",
                "text": json.dumps(
                    {
                        "verbatimDate": str(
                            row["verbatimDate"]
                        ),
                        "verbatimLocality": str(
                            row["verbatimLocality"]
                        ),
                    },
                    ensure_ascii=False,
                ),
            })

    # Actual test image
    content.append({
        "type": "text",
        "text": USER_PROMPT_TEMPLATE,
    })

    content.append({
        "type": "image",
        "image": str(image_path),
    })

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ---------------------------------------------------------------------
# JSON PARSING
# ---------------------------------------------------------------------

def clean_json_text(text: str):
    """
    Extract JSON from a VLM response.

    Handles:
      - plain JSON
      - ```json ... ```
      - surrounding commentary
      - truncated-looking responses where possible
    """

    if not text:
        raise ValueError("Empty model response")

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response:\n{text}"
    )


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize_result(parsed: dict):

    date = parsed.get("verbatimDate", "MISSING")
    locality = parsed.get("verbatimLocality", "MISSING")

    date_conf = parsed.get(
        "date_confidence",
        0.5,
    )

    loc_conf = parsed.get(
        "locality_confidence",
        0.5,
    )

    # Safety against null/empty values
    if date is None or str(date).strip() == "":
        date = "MISSING"

    if locality is None or str(locality).strip() == "":
        locality = "MISSING"

    try:
        date_conf = float(date_conf)
    except Exception:
        date_conf = 0.5

    try:
        loc_conf = float(loc_conf)
    except Exception:
        loc_conf = 0.5

    date_conf = max(0.0, min(1.0, date_conf))
    loc_conf = max(0.0, min(1.0, loc_conf))

    return {
        "verbatimDate": str(date).strip(),
        "verbatimLocality": str(locality).strip(),
        "date_confidence": date_conf,
        "locality_confidence": loc_conf,
    }


# ---------------------------------------------------------------------
# MODEL INFERENCE
# ---------------------------------------------------------------------

def transcribe_one(
    model,
    processor,
    image_path: Path,
    fewshot_examples,
    images_dir: Path,
):
    """
    Run one image through Qwen2.5-VL.
    """

    messages = build_messages(
        image_path,
        fewshot_examples,
        images_dir,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
        )

    # Remove input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    parsed = clean_json_text(output_text)

    return normalize_result(parsed)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV containing image_file",
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Directory containing images",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV",
    )

    parser.add_argument(
        "--train-csv",
        required=True,
        help="Training CSV used for few-shot examples",
    )

    parser.add_argument(
        "--n-fewshot",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N images",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------------

    if torch.cuda.is_available():
        device = "cuda"
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
    else:
        device = "cpu"
        print(
            "WARNING: CUDA unavailable. "
            "This will be extremely slow."
        )

    print("=" * 70)
    print("Local VLM inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------------

    images_dir = Path(args.images)

    df = pd.read_csv(args.csv)
    train_df = pd.read_csv(args.train_csv)

    if args.limit is not None:
        df = df.head(args.limit)

    print(f"Input rows: {len(df)}")
    print(f"Images directory: {images_dir}")

    # ---------------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------------

    print("\nLoading Qwen2.5-VL-3B-Instruct...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    print("Model loaded successfully.")

    # ---------------------------------------------------------------
    # FEW SHOT
    # ---------------------------------------------------------------

    fewshot_examples = build_fewshot_examples(
        train_df,
        images_dir,
        args.n_fewshot,
    )

    print(
        f"Few-shot examples: "
        f"{len(fewshot_examples)}"
    )

    # ---------------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------------

    rows = []

    for i, row in df.iterrows():

        image_file = row["image_file"]
        image_path = images_dir / image_file

        try:

            check_image(image_path)

            result = transcribe_one(
                model=model,
                processor=processor,
                image_path=image_path,
                fewshot_examples=fewshot_examples,
                images_dir=images_dir,
            )

            rows.append({
                "image_file": image_file,
                "verbatimDate": result[
                    "verbatimDate"
                ],
                "verbatimLocality": result[
                    "verbatimLocality"
                ],
                "date_confidence_raw": result[
                    "date_confidence"
                ],
                "locality_confidence_raw": result[
                    "locality_confidence"
                ],
            })

            print(
                f"[{i + 1}/{len(df)}] "
                f"{image_file} -> "
                f"date={result['verbatimDate']!r} "
                f"loc={result['verbatimLocality']!r} "
                f"date_conf={result['date_confidence']:.2f} "
                f"loc_conf={result['locality_confidence']:.2f}"
            )

        except Exception as e:

            print(
                f"[{i + 1}/{len(df)}] "
                f"{image_file} FAILED: {e}"
            )

            rows.append({
                "image_file": image_file,
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence_raw": 0.0,
                "locality_confidence_raw": 0.0,
            })

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    out_df = pd.DataFrame(rows)

    out_df.to_csv(
        args.out,
        index=False,
    )

    print("\n" + "=" * 70)
    print(
        f"Wrote {len(out_df)} rows to:"
    )
    print(args.out)
    print("=" * 70)


if __name__ == "__main__":
    main()