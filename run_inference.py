"""
Local VLM inference pipeline for Danish museum specimen-label
transcription.

Model:
    Qwen/Qwen2.5-VL-3B-Instruct

Designed for:
    Kaggle Tesla T4

Pipeline:
    Original image
        -> single-pass resize directly to the model's target pixel budget
           (replaces the old "upscale 4x, then let the processor silently
           downsample again" double-resample, which was blurring fine
           handwriting)
        -> mild contrast + sharpness enhancement
        -> Qwen2.5-VL (prompt now actually imported from prompt_template.py)
        -> JSON parse + normalize
        -> optional lightweight verification pass on low-confidence fields
        -> CSV

Changes vs previous version (see chat for full diagnosis):
    1. FIXED: prompt_template.py is now actually imported and used. The
       previous version defined its own inline copy of the prompt and
       silently ignored prompt_template.py.
    2. Image preprocessing no longer double-resamples. Upscale factor is
       now a *minimum enlargement floor*, and the final size is computed
       once, directly against min_pixels/max_pixels.
    3. Few-shot selection defaults to 3 examples (not 4) with a fairer mix
       (1 multi-card, 1 fully-populated, 1 partially-MISSING) instead of
       stacking 3 different flavors of MISSING example, which was biasing
       the model toward outputting MISSING.
    4. Few-shot images use a smaller pixel budget than the target image --
       they're there to teach formatting conventions, not for fine OCR.
    5. Added --verify: an optional second pass that only fires for fields
       with confidence < 0.5, re-checking that specific field against the
       image. Off by default to keep runtime predictable; turn on once the
       base pipeline looks reasonable.
    6. Periodic torch.cuda.empty_cache() and lower default resolution
       ceiling headroom so 3-fewshot + verify runs don't creep toward OOM.
"""


import argparse
import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch

from PIL import Image, ImageEnhance

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info

from prompt_template import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    FEWSHOT_INSTRUCTION,
)


# ================================================================
# MODEL
# ================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ================================================================
# IMAGE PREPROCESSING SETTINGS
# ================================================================

# Minimum enlargement floor for small source images. This is NOT a fixed
# multiplier applied blindly -- the actual final size is always computed
# directly against TARGET_MIN_PIXELS/TARGET_MAX_PIXELS below in a single
# resize pass, so this only matters for images small enough that 1x
# wouldn't reach TARGET_MIN_PIXELS.
UPSCALE_FLOOR = 2

CONTRAST_FACTOR = 1.10
SHARPNESS_FACTOR = 1.25

# Target image (the one being transcribed): kept at the same ceiling as
# the previous T4-safe configuration.
TARGET_MIN_PIXELS = 384 * 28 * 28   # ~301k px
TARGET_MAX_PIXELS = 1024 * 28 * 28  # ~803k px

# Few-shot example images: lower budget. These exist to demonstrate
# formatting conventions (date style, " | " usage, MISSING usage), not for
# fine-grained OCR of the example itself, so they don't need full
# resolution. This also directly reduces per-image VRAM/token cost when
# using multiple few-shot examples.
FEWSHOT_MIN_PIXELS = 256 * 28 * 28  # ~201k px
FEWSHOT_MAX_PIXELS = 448 * 28 * 28  # ~351k px


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

def smart_resize_image(
    image_path: Path,
    min_pixels: int,
    max_pixels: int,
    upscale_floor: int = UPSCALE_FLOOR,
):
    """
    Load and resize an image ONCE, directly to the pixel budget the model
    will actually use.

    Previous versions upscaled 4x unconditionally, then relied on Qwen's
    AutoProcessor to downsample again to fit min_pixels/max_pixels. That
    is two successive LANCZOS resamples, which introduces visible
    ringing/blur on fine handwriting -- a very plausible cause of digit/
    letter confusions and misreads observed in testing.

    This version computes the final target size directly:
      - Start from upscale_floor x original size.
      - If that's above max_pixels, scale down to fit max_pixels.
      - If that's below min_pixels, scale up to fit min_pixels.
      - Resize once.
    """

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not load image {image_path}: {e}")

    w, h = img.size

    target_w = w * upscale_floor
    target_h = h * upscale_floor
    target_pixels = target_w * target_h

    if target_pixels > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
    elif target_pixels < min_pixels:
        scale = (min_pixels / (w * h)) ** 0.5
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(CONTRAST_FACTOR)
    img = ImageEnhance.Sharpness(img).enhance(SHARPNESS_FACTOR)

    return img


def check_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise RuntimeError(f"Could not read image {path}: {e}")


# ================================================================
# FEW-SHOT SELECTION
# ================================================================

def build_fewshot_examples(train_df: pd.DataFrame, n: int, seed: int = 0):
    """
    Pick a small, DIVERSE set of few-shot examples: one multi-card example
    (teaches " | " usage), one fully-populated "clean" example (teaches
    what a confident, present-text answer looks like -- this was largely
    absent from the previous selection, which stacked up to 3 different
    flavors of MISSING and likely biased the model toward over-hedging),
    and one partially-MISSING example (teaches legitimate MISSING usage
    without over-representing it). Remaining slots (if n > 3) filled
    randomly.
    """

    train_df = train_df.copy()
    selected = []
    used = set()

    def add(row):
        if row["image_file"] not in used:
            selected.append(row)
            used.add(row["image_file"])

    # Multi-card example.
    pipe_mask = (
        train_df["verbatimDate"].astype(str).str.contains(r"\|", na=False)
        | train_df["verbatimLocality"].astype(str).str.contains(r"\|", na=False)
    )
    pipe_rows = train_df[pipe_mask]
    if len(pipe_rows) > 0:
        add(pipe_rows.sample(1, random_state=seed).iloc[0])

    # Fully-populated "clean" example -- both fields present, single card.
    clean_rows = train_df[
        ~train_df["verbatimDate"].astype(str).eq("MISSING")
        & ~train_df["verbatimLocality"].astype(str).eq("MISSING")
        & ~pipe_mask
    ]
    if len(clean_rows) > 0:
        add(clean_rows.sample(1, random_state=seed + 1).iloc[0])

    # One partially-MISSING example (either field, not both -- avoid
    # teaching "both fields missing" as a common pattern).
    partial_missing_rows = train_df[
        train_df["verbatimDate"].astype(str).eq("MISSING")
        ^ train_df["verbatimLocality"].astype(str).eq("MISSING")
    ]
    if len(partial_missing_rows) > 0:
        add(partial_missing_rows.sample(1, random_state=seed + 2).iloc[0])

    # Fill any remaining requested slots randomly.
    remaining = train_df[~train_df["image_file"].isin(used)]
    while len(selected) < n and len(remaining) > 0:
        row = remaining.sample(1, random_state=seed + len(selected) + 20).iloc[0]
        add(row)
        remaining = remaining[remaining["image_file"] != row["image_file"]]

    return selected[:n]


# ================================================================
# BUILD MULTIMODAL MESSAGES
# ================================================================

def build_messages(image_path: Path, fewshot_examples, images_dir: Path):

    content = [{"type": "text", "text": SYSTEM_PROMPT}]

    if fewshot_examples:
        content.append({
            "type": "text",
            "text": FEWSHOT_INSTRUCTION.format(n=len(fewshot_examples)),
        })

        for row in fewshot_examples:
            example_path = images_dir / str(row["image_file"])
            if not example_path.exists():
                continue

            example_img = smart_resize_image(
                example_path,
                min_pixels=FEWSHOT_MIN_PIXELS,
                max_pixels=FEWSHOT_MAX_PIXELS,
            )

            content.append({"type": "image", "image": example_img})

            answer = {
                "verbatimDate": str(row["verbatimDate"]),
                "verbatimLocality": str(row["verbatimLocality"]),
            }
            content.append({
                "type": "text",
                "text": json.dumps(answer, ensure_ascii=False),
            })

    content.append({"type": "text", "text": USER_PROMPT_TEMPLATE})

    target_img = smart_resize_image(
        image_path,
        min_pixels=TARGET_MIN_PIXELS,
        max_pixels=TARGET_MAX_PIXELS,
    )
    content.append({"type": "image", "image": target_img})

    return [{"role": "user", "content": content}]


def build_verify_messages(image_path: Path, field_name: str, current_value: str):
    """
    Lightweight single-field verification pass. Only called for fields
    with confidence < 0.5 that are non-MISSING. Re-shows the (already
    correctly-sized) target image and asks the model to specifically
    re-check one field, without the full few-shot context -- this keeps
    the extra pass cheap.
    """

    label = "date" if field_name == "verbatimDate" else "locality"

    prompt = f"""Look closely at this specimen label image again, specifically at the {label}.

Your previous answer for {label} was: "{current_value}"

Trace every character carefully. If this transcription is correct, repeat it
exactly. If it is wrong, provide the corrected exact transcription. If the
{label} is not actually legible, respond with MISSING.

Respond with ONLY this JSON, nothing else:
{{"value": "<corrected string or MISSING>", "confidence": 0.0}}
"""

    target_img = smart_resize_image(
        image_path,
        min_pixels=TARGET_MIN_PIXELS,
        max_pixels=TARGET_MAX_PIXELS,
    )

    content = [
        {"type": "image", "image": target_img},
        {"type": "text", "text": prompt},
    ]

    return [{"role": "user", "content": content}]


# ================================================================
# JSON PARSING
# ================================================================

def clean_json_text(text: str):
    if not text:
        raise ValueError("Empty model response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from model response:\n" + text)


# ================================================================
# NORMALIZATION
# ================================================================

def normalize_result(parsed: dict):
    date = parsed.get("verbatimDate", "MISSING")
    locality = parsed.get("verbatimLocality", "MISSING")
    date_conf = parsed.get("date_confidence", 0.0)
    loc_conf = parsed.get("locality_confidence", 0.0)

    if date is None or str(date).strip() == "":
        date = "MISSING"
    if locality is None or str(locality).strip() == "":
        locality = "MISSING"

    try:
        date_conf = float(date_conf)
    except Exception:
        date_conf = 0.0
    try:
        loc_conf = float(loc_conf)
    except Exception:
        loc_conf = 0.0

    date_conf = max(0.0, min(1.0, date_conf))
    loc_conf = max(0.0, min(1.0, loc_conf))

    if str(date).strip() == "MISSING":
        date_conf = min(date_conf, 0.05)
    if str(locality).strip() == "MISSING":
        loc_conf = min(loc_conf, 0.05)

    return {
        "verbatimDate": str(date).strip(),
        "verbatimLocality": str(locality).strip(),
        "date_confidence": date_conf,
        "locality_confidence": loc_conf,
    }


# ================================================================
# GENERATION HELPER
# ================================================================

def run_generation(model, processor, messages, max_new_tokens: int):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.03,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del inputs, generated_ids, generated_ids_trimmed
    return output_text


# ================================================================
# MODEL INFERENCE (per image)
# ================================================================

def transcribe_one(
    model,
    processor,
    image_path: Path,
    fewshot_examples,
    images_dir: Path,
    verify: bool,
    verify_threshold: float = 0.5,
):

    messages = build_messages(image_path, fewshot_examples, images_dir)
    output_text = run_generation(model, processor, messages, max_new_tokens=180)
    parsed = clean_json_text(output_text)
    result = normalize_result(parsed)

    if verify:
        for field, conf_key in (
            ("verbatimDate", "date_confidence"),
            ("verbatimLocality", "locality_confidence"),
        ):
            value = result[field]
            conf = result[conf_key]

            if value == "MISSING" or conf >= verify_threshold:
                continue

            try:
                v_messages = build_verify_messages(image_path, field, value)
                v_output = run_generation(model, processor, v_messages, max_new_tokens=60)
                v_parsed = clean_json_text(v_output)

                v_value = v_parsed.get("value", value)
                v_conf = v_parsed.get("confidence", conf)

                if v_value is None or str(v_value).strip() == "":
                    v_value = "MISSING"
                try:
                    v_conf = float(v_conf)
                except Exception:
                    v_conf = conf
                v_conf = max(0.0, min(1.0, v_conf))
                if str(v_value).strip() == "MISSING":
                    v_conf = min(v_conf, 0.05)

                result[field] = str(v_value).strip()
                result[conf_key] = v_conf

            except Exception:
                # Verification failures should never crash the main
                # pipeline -- keep the original (unverified) result.
                pass

    return result


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV containing image_file")
    parser.add_argument("--images", required=True, help="Directory containing images")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.add_argument("--train-csv", required=True, help="Training CSV for few-shot examples")
    parser.add_argument("--n-fewshot", type=int, default=3,
                         help="Default lowered from 4 to 3 -- see diagnosis notes.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N images")
    parser.add_argument("--verify", action="store_true",
                         help="Enable second-pass verification for low-confidence "
                              "non-MISSING fields (confidence < --verify-threshold). "
                              "Roughly doubles cost for the affected fields only.")
    parser.add_argument("--verify-threshold", type=float, default=0.5)
    parser.add_argument("--empty-cache-every", type=int, default=10,
                         help="Call torch.cuda.empty_cache() every N images.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("WARNING: CUDA unavailable.")

    print("=" * 70)
    print("Local VLM inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Verify pass: {args.verify} (threshold={args.verify_threshold})")
    print("=" * 70)

    images_dir = Path(args.images)
    df = pd.read_csv(args.csv)
    train_df = pd.read_csv(args.train_csv)

    if args.limit is not None:
        df = df.head(args.limit)

    print(f"Input rows: {len(df)}")
    print(f"Images directory: {images_dir}")

    print("\nLoading Qwen2.5-VL-3B-Instruct...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=TARGET_MIN_PIXELS,
        max_pixels=TARGET_MAX_PIXELS,
    )

    print("Model loaded successfully.")
    print("Visual preprocessing (single-pass resize, no double-resample):")
    print(f"  Target image: min_pixels={TARGET_MIN_PIXELS} max_pixels={TARGET_MAX_PIXELS}")
    print(f"  Few-shot images: min_pixels={FEWSHOT_MIN_PIXELS} max_pixels={FEWSHOT_MAX_PIXELS}")
    print(f"  Upscale floor: {UPSCALE_FLOOR}x  Contrast: {CONTRAST_FACTOR}  Sharpness: {SHARPNESS_FACTOR}")

    fewshot_examples = build_fewshot_examples(train_df, args.n_fewshot)
    print(f"Few-shot examples: {len(fewshot_examples)}")
    for example in fewshot_examples:
        print("  Example:", example["image_file"], "->",
              example["verbatimDate"], "|", example["verbatimLocality"])

    rows = []

    for i, row in df.iterrows():
        image_file = row["image_file"]
        image_path = images_dir / str(image_file)

        try:
            check_image(image_path)

            result = transcribe_one(
                model=model,
                processor=processor,
                image_path=image_path,
                fewshot_examples=fewshot_examples,
                images_dir=images_dir,
                verify=args.verify,
                verify_threshold=args.verify_threshold,
            )

            rows.append({
                "image_file": image_file,
                "verbatimDate": result["verbatimDate"],
                "verbatimLocality": result["verbatimLocality"],
                "date_confidence_raw": result["date_confidence"],
                "locality_confidence_raw": result["locality_confidence"],
            })

            print(
                f"[{i + 1}/{len(df)}] {image_file} -> "
                f"date={result['verbatimDate']!r} "
                f"loc={result['verbatimLocality']!r} "
                f"date_conf={result['date_confidence']:.2f} "
                f"loc_conf={result['locality_confidence']:.2f}"
            )

        except Exception as e:
            print(f"[{i + 1}/{len(df)}] {image_file} FAILED: {e}")
            rows.append({
                "image_file": image_file,
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence_raw": 0.0,
                "locality_confidence_raw": 0.0,
            })

        if torch.cuda.is_available() and (i + 1) % args.empty_cache_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print("\n" + "=" * 70)
    print(f"Wrote {len(out_df)} rows to:")
    print(args.out)
    print("=" * 70)


if __name__ == "__main__":
    main()