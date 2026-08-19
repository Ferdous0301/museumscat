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

from image_utils import (
    smart_resize_image,
    check_image,
    UPSCALE_FLOOR,
    CONTRAST_FACTOR,
    SHARPNESS_FACTOR,
    TARGET_MIN_PIXELS,
    TARGET_MAX_PIXELS,
    FEWSHOT_MIN_PIXELS,
    FEWSHOT_MAX_PIXELS,
    ESCALATED_MAX_PIXELS,
)


# ================================================================
# MODEL
# ================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# Image preprocessing (resize/contrast/sharpen, pixel budgets, and
# check_image) now live in image_utils.py, shared with ocr_crosscheck.py.
# See that module for the "why" behind the single-pass resize approach.


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

def build_messages(image_path: Path, fewshot_examples, images_dir: Path,
                    target_max_pixels: int = TARGET_MAX_PIXELS):

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
        max_pixels=target_max_pixels,
    )
    content.append({"type": "image", "image": target_img})

    return [{"role": "user", "content": content}]


def build_verify_messages(image_path: Path, field_name: str, current_value: str):
    """
    Lightweight single-field verification pass. Re-shows the (already
    correctly-sized) target image and asks the model to specifically
    re-check one field, without the full few-shot context -- this keeps
    the extra pass cheap.

    Two modes, chosen by whether current_value is MISSING:
      - "double check": the model previously gave a non-MISSING answer at
        low confidence. Ask it to re-verify that specific transcription.
      - "second look": the model previously said MISSING. Ask it to look
        again specifically for any text it may have missed, rather than
        re-confirming a value (there is no value to re-confirm). This
        matters because a model that says MISSING is not necessarily
        right -- it may have simply not looked hard enough at faint or
        small text, which is a different failure mode than misreading
        something it did find.
    """

    label = "date" if field_name == "verbatimDate" else "locality"

    if current_value == "MISSING":
        prompt = f"""Look very closely at this specimen label image again, specifically
searching for the {label}.

Your previous answer for {label} was MISSING. Before confirming that, check:
- small or faint handwriting near the edges of the card
- text partially obscured by the specimen itself
- a second physical card elsewhere in the image

If you can now make out a {label}, transcribe it exactly. If you genuinely
cannot find any {label} text after this closer look, confirm MISSING.

Respond with ONLY this JSON, nothing else:
{{"value": "<string or MISSING>", "confidence": 0.0}}
"""
    else:
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
# METADATA GUARDRAIL
# ================================================================

# Observed on real runs: the model sometimes outputs collector/museum
# metadata as locality (e.g. "Coll. J. P. Johansen", "Coll. Schjødtæ",
# "Mus. Lev.") even at self-reported confidence 0.00, despite the prompt
# explicitly forbidding this. A 3B model will not perfectly self-censor
# every case through instructions alone, so this is enforced deterministically
# after generation rather than relying only on the prompt.
#
# This is intentionally a narrow, high-precision blocklist -- it only
# strips text that STARTS WITH (or IS ENTIRELY) a known metadata marker, or
# matches an "Initial. Initial. Surname" personal-name pattern. It does not
# touch legitimate short place abbreviations (e.g. "Ti", "Kb", "Bovbj.")
# since none of those match these patterns. If the field is stripped, the
# whole field becomes MISSING (not just the metadata token) -- see
# strip_metadata_locality docstring for why partial-strip is intentionally
# NOT attempted.

_METADATA_PREFIX_RE = re.compile(
    r"^\s*(coll\.?|collector|det\.?|determin\w*|dania|mus\.?|museum|tilg\.?|"
    r"acc\.?|accession|cat\.?|catalog\w*)\b",
    re.IGNORECASE,
)

# "O. Mic. Hansen" / "J. P. Johansen" style: one or more single-letter (or
# short) initials followed by a capitalized surname, and nothing else.
_PERSON_NAME_RE = re.compile(
    r"^\s*(?:[A-ZÆØÅ]\.?\s*){1,3}[A-ZÆØÅ][a-zæøåé]+\s*\.?\s*$"
)


def strip_metadata_locality(locality: str) -> tuple[str, bool]:
    """
    Returns (possibly-replaced locality, was_stripped).

    We deliberately replace the ENTIRE field with MISSING rather than trying
    to strip just the metadata token and keep any trailing text. Partial
    stripping would require confidently identifying where metadata ends and
    a real place name begins within a short string produced by a 3B model --
    that's exactly the kind of judgment call that just failed. Falling back
    to MISSING is consistent with the task's own stated preference
    ("MISSING is always better than a wrong/unsupported answer").

    Does not touch " | "-joined multi-card values beyond checking each card
    segment independently, so a real locality in another card is preserved.
    """
    if not isinstance(locality, str) or locality.strip().upper() in ("MISSING", ""):
        return locality, False

    cards = locality.split("|")
    kept_cards = []
    any_stripped = False

    for card in cards:
        card_stripped = card.strip()
        if _METADATA_PREFIX_RE.match(card_stripped) or _PERSON_NAME_RE.match(card_stripped):
            any_stripped = True
            continue
        kept_cards.append(card_stripped)

    if not kept_cards:
        return "MISSING", any_stripped

    return " | ".join(kept_cards), any_stripped


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

    locality, was_stripped = strip_metadata_locality(str(locality).strip())

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
        "locality_metadata_stripped": was_stripped,
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
    verify_missing: bool = False,
    escalate: bool = False,
):

    messages = build_messages(image_path, fewshot_examples, images_dir)
    output_text = run_generation(model, processor, messages, max_new_tokens=180)
    parsed = clean_json_text(output_text)
    result = normalize_result(parsed)

    # Tracked separately from the main result dict so they're easy to add as
    # extra (non-breaking) CSV columns -- risk_ranking.py uses these as a
    # signal: a field that changed under a second, differently-worded look
    # is less trustworthy than one the model confirmed unchanged.
    result["date_verify_changed"] = False
    result["locality_verify_changed"] = False

    if verify:
        for field, conf_key, changed_key in (
            ("verbatimDate", "date_confidence", "date_verify_changed"),
            ("verbatimLocality", "locality_confidence", "locality_verify_changed"),
        ):
            value = result[field]
            conf = result[conf_key]

            is_missing = (value == "MISSING")

            if is_missing and not verify_missing:
                continue
            if not is_missing and conf >= verify_threshold:
                continue

            v_output = None
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

                v_value = str(v_value).strip()
                if field == "verbatimLocality":
                    v_value, _ = strip_metadata_locality(v_value)

                if v_value == "MISSING":
                    v_conf = min(v_conf, 0.05)

                changed = (v_value != value)
                result[field] = v_value
                result[conf_key] = v_conf
                result[changed_key] = changed
                print(f"    [verify {field} on {image_path.name}: "
                      f"{'CHANGED' if changed else 'confirmed'} "
                      f"{value!r} -> {v_value!r} (conf {conf:.2f} -> {v_conf:.2f})]")

            except Exception as e:
                # Verification failures should never crash the main
                # pipeline, but silently swallowing them makes it
                # impossible to tell "verify ran and confirmed the
                # original answer" apart from "verify never actually
                # ran". Log it instead so that distinction is visible.
                print(f"    [verify FAILED for {field} on {image_path.name}: {e}] "
                      f"raw output: {v_output!r}")

    if escalate and result["verbatimDate"] == "MISSING" and result["verbatimLocality"] == "MISSING":
        try:
            esc_messages = build_messages(
                image_path, fewshot_examples, images_dir,
                target_max_pixels=ESCALATED_MAX_PIXELS,
            )
            esc_output = run_generation(model, processor, esc_messages, max_new_tokens=180)
            esc_parsed = clean_json_text(esc_output)
            esc_result = normalize_result(esc_parsed)

            found_something = (
                esc_result["verbatimDate"] != "MISSING"
                or esc_result["verbatimLocality"] != "MISSING"
            )
            print(f"    [escalated resolution retry on {image_path.name}: "
                  f"date={esc_result['verbatimDate']!r} loc={esc_result['verbatimLocality']!r} "
                  f"({'found new text' if found_something else 'still MISSING -- likely a genuine resolution ceiling for this image'})]")

            if found_something:
                result = esc_result

        except Exception as e:
            print(f"    [escalation FAILED on {image_path.name}: {e}]")

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
    parser.add_argument("--verify-missing", action="store_true",
                         help="With --verify, also take a second look at fields "
                              "the model reported as MISSING, in case it simply "
                              "missed small/faint text rather than the field "
                              "genuinely being absent. Off by default since it "
                              "adds cost for every MISSING field, not just "
                              "low-confidence ones.")
    parser.add_argument("--escalate", action="store_true",
                         help="If both fields are still MISSING after the normal "
                              "pass (and verify, if enabled), retry that one image "
                              "once at a higher resolution "
                              f"(max_pixels={ESCALATED_MAX_PIXELS}, still well under "
                              "the known T4 OOM boundary). Only fires for hard cases, "
                              "so cost impact should be small unless most of your "
                              "images are hard cases.")
    parser.add_argument("--empty-cache-every", type=int, default=10,
                         help="Call torch.cuda.empty_cache() every N images.")
    parser.add_argument("--lora-adapter", type=str, default=None,
                         help="Path to a LoRA adapter directory produced by "
                              "train_lora.py. If given, it's loaded on top of "
                              "the base model via peft. Requires 'pip install peft'.")
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
    print(f"Verify pass: {args.verify} (threshold={args.verify_threshold}, verify_missing={args.verify_missing})")
    print(f"Escalate on double-MISSING: {args.escalate} (max_pixels={ESCALATED_MAX_PIXELS})")
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

    if args.lora_adapter:
        try:
            from peft import PeftModel
        except ImportError:
            raise RuntimeError(
                "--lora-adapter was given but peft is not installed. "
                "Run: pip install peft --break-system-packages"
            )
        print(f"Loading LoRA adapter from {args.lora_adapter} ...")
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        print("LoRA adapter loaded.")

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
                verify_missing=args.verify_missing,
                escalate=args.escalate,
            )

            rows.append({
                "image_file": image_file,
                "verbatimDate": result["verbatimDate"],
                "verbatimLocality": result["verbatimLocality"],
                "date_confidence_raw": result["date_confidence"],
                "locality_confidence_raw": result["locality_confidence"],
                # Extra columns, additive only -- the 5 columns above match
                # the required submission format exactly. These feed
                # risk_ranking.py and ocr_crosscheck.py; safe to ignore
                # otherwise.
                "date_verify_changed": result.get("date_verify_changed", False),
                "locality_verify_changed": result.get("locality_verify_changed", False),
            })

            stripped_note = " [metadata stripped]" if result.get("locality_metadata_stripped") else ""
            print(
                f"[{i + 1}/{len(df)}] {image_file} -> "
                f"date={result['verbatimDate']!r} "
                f"loc={result['verbatimLocality']!r} "
                f"date_conf={result['date_confidence']:.2f} "
                f"loc_conf={result['locality_confidence']:.2f}"
                f"{stripped_note}"
            )

        except Exception as e:
            print(f"[{i + 1}/{len(df)}] {image_file} FAILED: {e}")
            rows.append({
                "image_file": image_file,
                "verbatimDate": "MISSING",
                "verbatimLocality": "MISSING",
                "date_confidence_raw": 0.0,
                "locality_confidence_raw": 0.0,
                "date_verify_changed": False,
                "locality_verify_changed": False,
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