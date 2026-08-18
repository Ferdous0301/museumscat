"""
Local VLM inference pipeline for the Danish Dung Beetle label
transcription task.

Model:
    Qwen/Qwen2.5-VL-3B-Instruct

Designed for:
    Kaggle Tesla T4 GPU

Main goals:
    - Exact visual transcription
    - High-resolution label reading
    - Low hallucination
    - Correct MISSING handling
    - Multi-card specimen handling
    - Danish locality preservation
    - Conservative confidence scores
    - T4-friendly VRAM usage
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ================================================================
# MODEL
# ================================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ================================================================
# VISUAL RESOLUTION
# ================================================================

# Qwen VL uses 28x28 visual patches.
#
# Higher max_pixels = better ability to read tiny text,
# but also considerably more VRAM.
#
# These values are deliberately chosen for a Tesla T4.
MIN_PIXELS = 512 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


# ================================================================
# PROMPT
# ================================================================

SYSTEM_PROMPT = r"""
You are performing HIGH-PRECISION OCR and EXACT VISUAL TRANSCRIPTION
of historical museum specimen labels.

Your task is to read the text that is VISIBLY PRESENT in the image.

You must extract exactly TWO fields:

1. verbatimDate
2. verbatimLocality

This is a TRANSCRIPTION task.

It is NOT a classification task.
It is NOT a species-identification task.
It is NOT a geographic inference task.

============================================================
ABSOLUTE RULE: NEVER GUESS
============================================================

Only output information that you can actually see in the image.

NEVER:

- invent text
- infer a date
- infer a locality
- identify a place from background knowledge
- use the species to guess the locality
- use the collector to guess the locality
- use museum metadata as a locality
- silently correct unclear handwriting
- replace an uncertain character with a plausible character
- convert a historical date into a modern date format

If a field cannot be read reliably from visible text, output:

"MISSING"

A correct MISSING is ALWAYS preferable to a guessed answer.

============================================================
WHAT TO LOOK FOR
============================================================

Inspect the ENTIRE IMAGE.

Museum specimen images may contain:

- one specimen label
- several physical labels/cards
- handwritten labels
- printed labels
- small labels
- overlapping cards
- collection labels
- determination labels
- museum/catalog information

You must distinguish COLLECTION INFORMATION from unrelated metadata.

The two target fields are:

DATE:
The date associated with the specimen collection.

LOCALITY:
The geographic place where the specimen was collected.

============================================================
DATE TRANSCRIPTION
============================================================

Transcribe the collection date EXACTLY as visible.

Preserve:

- punctuation
- spaces
- slashes
- dots
- hyphens
- Roman numerals
- Arabic numerals
- abbreviations
- capitalization
- historical formatting

Examples:

27.IV.2022
7/6 1870
1.7.2000
22.5.1977.
Juli 1930
15.V.2011

DO NOT normalize dates.

For example:

Visible:
27.IV.2022

Correct:
27.IV.2022

Incorrect:
27.04.2022

Incorrect:
27/04/2022

If the label visibly contains multiple distinct collection dates,
include all relevant dates in their visible order, separated by:

" | "

Example:

"5/2 53 | 22-9-36"

Do NOT include dates that clearly belong to:

- determination
- cataloging
- accession
- later museum processing
- unrelated metadata

============================================================
LOCALITY TRANSCRIPTION
============================================================

Transcribe the geographic collection locality EXACTLY as visible.

Preserve:

- spelling
- capitalization
- abbreviations
- punctuation
- Danish letters
- historical spelling
- locality hierarchy

Examples:

Dyrehaven
Tisvilde
Lodskovvad
Svinø strand
Bovbj.
Røsnæsgd. NWZ
Kb

Do NOT translate Danish place names.

Do NOT expand abbreviations.

Do NOT modernize spelling.

============================================================
SHORT LOCALITY ABBREVIATIONS
============================================================

Short text can still be a valid locality.

For example:

"Ti"
"Kb"

may be valid locality annotations if the label clearly uses them
as geographic collection information.

Therefore:

DO NOT automatically discard short words.

However, do not turn arbitrary museum metadata into locality.

Use the visual context of the label.

============================================================
DANIA / MUSEUM METADATA
============================================================

"Dania" is NEVER a locality.

If "Dania" appears as collection or museum information,
do NOT output it as verbatimLocality.

If there is no other visible geographic locality:

verbatimLocality = "MISSING"

Similarly, do not treat the following as locality unless the image
clearly shows that they are geographic collection information:

- collection names
- museum names
- catalog numbers
- collector names
- determination information
- "det."
- "Coll."
- accession information
- catalog metadata

============================================================
HABITAT AND SUBSTRATE
============================================================

Habitat or substrate descriptions are NOT localities.

For example:

"i kogødning"

describes habitat/substrate.

Do NOT output it as the locality.

Only output the geographic place.

============================================================
MULTIPLE PHYSICAL CARDS
============================================================

Some specimen photographs contain multiple physical cards.

You MUST inspect all visible cards before answering.

If multiple cards contain separate collection records,
extract all relevant values.

Separate multiple values with:

" | "

Example:

verbatimDate:
"5/2 53 | 22-9-36"

Example:

verbatimLocality:
"Place A | Place B"

Do NOT discard a second card simply because the first card already
contains a date or locality.

============================================================
HANDWRITING
============================================================

This is exact transcription.

Do not silently correct handwriting.

If a character visibly looks like:

5

write:

5

If it visibly looks like:

S

write:

S

Preserve visible Danish characters such as:

ø
æ
å

If a word contains one or more genuinely unreadable characters and
you cannot reliably determine the complete field, prefer:

"MISSING"

rather than inventing the word.

============================================================
IMPORTANT VISUAL PROCEDURE
============================================================

Before producing JSON:

1. Inspect the whole image.
2. Locate every physical card or label.
3. Read the smallest visible text carefully.
4. Identify candidate collection dates.
5. Identify candidate geographic localities.
6. Reject museum/catalog/collector metadata.
7. Reject habitat/substrate as locality.
8. Check every additional card.
9. Compare characters carefully.
10. Only then produce the final JSON.

Do not answer until the entire image has been inspected.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY one valid JSON object.

Do NOT use markdown.

Do NOT explain your answer.

Do NOT include reasoning.

Use exactly these four fields:

{
  "verbatimDate": "<string or MISSING>",
  "verbatimLocality": "<string or MISSING>",
  "date_confidence": 0.0,
  "locality_confidence": 0.0
}

============================================================
CONFIDENCE
============================================================

Confidence represents how clearly the requested text is visually readable.

0.95 - 1.00:
Very clear characters and almost no ambiguity.

0.80 - 0.94:
Readable with minor character uncertainty.

0.50 - 0.79:
Some ambiguity or difficult handwriting.

0.20 - 0.49:
Very difficult to read.

0.00 - 0.19:
Missing or not reliably readable.

IMPORTANT:

Do NOT assign high confidence simply because you found a plausible
answer.

If the text is unclear, lower the confidence.

If the field is MISSING, confidence must be close to 0.

If you are unsure whether a character is correct, prefer MISSING.
"""


USER_PROMPT = r"""
Inspect this specimen image at high visual attention.

Your ONLY task is to transcribe the visible COLLECTION DATE and
COLLECTION LOCALITY.

Before answering:

- inspect the entire image
- inspect every physical label/card
- look carefully at small handwritten text
- distinguish collection information from museum metadata
- preserve the exact visible spelling and punctuation
- preserve Danish characters
- preserve abbreviations
- do not normalize dates
- do not translate place names
- do not guess unreadable characters
- do not infer information from biological or geographic knowledge

If the collection date cannot be reliably read:

"verbatimDate": "MISSING"

If the collection locality cannot be reliably read:

"verbatimLocality": "MISSING"

Return ONLY the JSON object.
"""


FEWSHOT_INSTRUCTION = r"""
The following are examples from the same dataset.

Use them ONLY to learn the annotation conventions.

IMPORTANT:

The example answers are ground-truth annotations for the example
images only.

NEVER copy an example value into the target answer unless the same
text is actually visible in the target image.

Pay particular attention to:

- exact date formatting
- exact locality spelling
- abbreviations
- MISSING values
- multiple-card examples
- separation of locality from museum metadata
"""


# ================================================================
# IMAGE CHECK
# ================================================================

def check_image(path: Path):
    """Verify that an image exists and can be opened."""

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    try:
        with Image.open(path) as img:
            img.verify()

    except Exception as e:
        raise RuntimeError(
            f"Could not read image {path}: {e}"
        )


# ================================================================
# FEW-SHOT SELECTION
# ================================================================

def build_fewshot_examples(
    train_df: pd.DataFrame,
    images_dir: Path,
    n: int,
    seed: int = 0,
):
    """
    Select useful examples.

    Priority:
        1. multi-card examples
        2. MISSING examples
        3. ordinary examples

    IMPORTANT:
        On a Tesla T4, keep n small.
        Recommended: 2.
    """

    train_df = train_df.copy()

    selected = []

    # ------------------------------------------------------------
    # MULTI-CARD EXAMPLE
    # ------------------------------------------------------------

    pipe_mask = (
        train_df["verbatimDate"]
        .astype(str)
        .str.contains(r"\|", na=False)
        |
        train_df["verbatimLocality"]
        .astype(str)
        .str.contains(r"\|", na=False)
    )

    pipe_rows = train_df[pipe_mask]

    if len(pipe_rows) > 0:

        selected.append(
            pipe_rows.sample(
                1,
                random_state=seed,
            ).iloc[0]
        )

    # ------------------------------------------------------------
    # MISSING EXAMPLE
    # ------------------------------------------------------------

    if len(selected) < n:

        missing_mask = (
            train_df["verbatimDate"]
            .astype(str)
            .eq("MISSING")
            |
            train_df["verbatimLocality"]
            .astype(str)
            .eq("MISSING")
        )

        missing_rows = train_df[missing_mask]

        if len(missing_rows) > 0:

            candidate = missing_rows.sample(
                1,
                random_state=seed + 1,
            ).iloc[0]

            if candidate["image_file"] not in {
                x["image_file"]
                for x in selected
            }:

                selected.append(candidate)

    # ------------------------------------------------------------
    # NORMAL EXAMPLES
    # ------------------------------------------------------------

    used = {
        x["image_file"]
        for x in selected
    }

    remaining = train_df[
        ~train_df["image_file"].isin(used)
    ]

    while len(selected) < n and len(remaining) > 0:

        row = remaining.sample(
            1,
            random_state=seed + len(selected) + 10,
        ).iloc[0]

        selected.append(row)

        remaining = remaining[
            remaining["image_file"] != row["image_file"]
        ]

    return selected[:n]


# ================================================================
# BUILD MULTIMODAL MESSAGES
# ================================================================

def build_messages(
    image_path: Path,
    fewshot_examples,
    images_dir: Path,
):

    content = []

    # Main instructions
    content.append({
        "type": "text",
        "text": SYSTEM_PROMPT,
    })

    # ------------------------------------------------------------
    # FEW-SHOT EXAMPLES
    # ------------------------------------------------------------

    if fewshot_examples:

        content.append({
            "type": "text",
            "text": FEWSHOT_INSTRUCTION,
        })

        for row in fewshot_examples:

            example_path = (
                images_dir /
                str(row["image_file"])
            )

            if not example_path.exists():
                continue

            content.append({
                "type": "image",
                "image": str(example_path),
            })

            answer = {
                "verbatimDate": str(
                    row["verbatimDate"]
                ),
                "verbatimLocality": str(
                    row["verbatimLocality"]
                ),
            }

            content.append({
                "type": "text",
                "text": json.dumps(
                    answer,
                    ensure_ascii=False,
                ),
            })

    # ------------------------------------------------------------
    # TARGET IMAGE
    # ------------------------------------------------------------

    content.append({
        "type": "text",
        "text": USER_PROMPT,
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


# ================================================================
# JSON PARSING
# ================================================================

def clean_json_text(text: str):

    if not text:
        raise ValueError(
            "Empty model response"
        )

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

    # Direct JSON
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(
                candidate
            )

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse JSON from model response:\n"
        + text
    )


# ================================================================
# NORMALIZATION
# ================================================================

def normalize_result(parsed: dict):

    date = parsed.get(
        "verbatimDate",
        "MISSING",
    )

    locality = parsed.get(
        "verbatimLocality",
        "MISSING",
    )

    date_conf = parsed.get(
        "date_confidence",
        0.0,
    )

    loc_conf = parsed.get(
        "locality_confidence",
        0.0,
    )

    # ------------------------------------------------------------
    # EMPTY VALUES
    # ------------------------------------------------------------

    if (
        date is None
        or str(date).strip() == ""
    ):
        date = "MISSING"

    if (
        locality is None
        or str(locality).strip() == ""
    ):
        locality = "MISSING"

    # ------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------

    try:
        date_conf = float(
            date_conf
        )
    except Exception:
        date_conf = 0.0

    try:
        loc_conf = float(
            loc_conf
        )
    except Exception:
        loc_conf = 0.0

    date_conf = max(
        0.0,
        min(1.0, date_conf),
    )

    loc_conf = max(
        0.0,
        min(1.0, loc_conf),
    )

    # Missing cannot have high confidence
    if date == "MISSING":
        date_conf = min(
            date_conf,
            0.05,
        )

    if locality == "MISSING":
        loc_conf = min(
            loc_conf,
            0.05,
        )

    return {
        "verbatimDate": str(
            date
        ).strip(),

        "verbatimLocality": str(
            locality
        ).strip(),

        "date_confidence": date_conf,

        "locality_confidence": loc_conf,
    }


# ================================================================
# MODEL INFERENCE
# ================================================================

def transcribe_one(
    model,
    processor,
    image_path: Path,
    fewshot_examples,
    images_dir: Path,
):

    messages = build_messages(
        image_path,
        fewshot_examples,
        images_dir,
    )

    # ------------------------------------------------------------
    # CHAT TEMPLATE
    # ------------------------------------------------------------

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # ------------------------------------------------------------
    # IMAGE PROCESSING
    # ------------------------------------------------------------

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device
    )

    # ------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,

            # JSON is very short.
            max_new_tokens=120,

            # Deterministic OCR/transcription.
            do_sample=False,

            # Slightly discourage repetitive output.
            repetition_penalty=1.05,
        )

    # ------------------------------------------------------------
    # REMOVE INPUT TOKENS
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # PARSE
    # ------------------------------------------------------------

    parsed = clean_json_text(
        output_text
    )

    return normalize_result(
        parsed
    )


# ================================================================
# MAIN
# ================================================================

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
        default=2,
        help="Number of few-shot examples. Use 2 on a Tesla T4.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N images",
    )

    args = parser.parse_args()

    # ============================================================
    # DEVICE
    # ============================================================

    if torch.cuda.is_available():

        device = "cuda"

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    else:

        device = "cpu"

        print(
            "WARNING: CUDA unavailable."
        )

    print("=" * 70)
    print("Local VLM inference")
    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Device: {device}"
    )
    print("=" * 70)

    # ============================================================
    # PATHS
    # ============================================================

    images_dir = Path(
        args.images
    )

    df = pd.read_csv(
        args.csv
    )

    train_df = pd.read_csv(
        args.train_csv
    )

    if args.limit is not None:

        df = df.head(
            args.limit
        )

    print(
        f"Input rows: {len(df)}"
    )

    print(
        f"Images directory: {images_dir}"
    )

    # ============================================================
    # MODEL
    # ============================================================

    print(
        "\nLoading Qwen2.5-VL-3B-Instruct..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )
    )

    # ============================================================
    # VISUAL RESOLUTION
    # ============================================================

    processor = AutoProcessor.from_pretrained(

        MODEL_NAME,

        min_pixels=MIN_PIXELS,

        max_pixels=MAX_PIXELS,
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Visual resolution:",
        f"min_pixels={MIN_PIXELS}",
        f"max_pixels={MAX_PIXELS}",
    )

    print(
        "Equivalent token grid:",
        "512 -> 1024 pixels",
    )

    # ============================================================
    # FEW-SHOT
    # ============================================================

    fewshot_examples = (
        build_fewshot_examples(
            train_df,
            images_dir,
            args.n_fewshot,
        )
    )

    print(
        f"Few-shot examples: "
        f"{len(fewshot_examples)}"
    )

    for example in fewshot_examples:

        print(
            "  Example:",
            example["image_file"],
            "->",
            example["verbatimDate"],
            "|",
            example["verbatimLocality"],
        )

    # ============================================================
    # INFERENCE
    # ============================================================

    rows = []

    for i, row in df.iterrows():

        image_file = row[
            "image_file"
        ]

        image_path = (
            images_dir /
            str(image_file)
        )

        try:

            check_image(
                image_path
            )

            result = transcribe_one(
                model=model,
                processor=processor,
                image_path=image_path,
                fewshot_examples=fewshot_examples,
                images_dir=images_dir,
            )

            rows.append({

                "image_file":
                    image_file,

                "verbatimDate":
                    result[
                        "verbatimDate"
                    ],

                "verbatimLocality":
                    result[
                        "verbatimLocality"
                    ],

                "date_confidence_raw":
                    result[
                        "date_confidence"
                    ],

                "locality_confidence_raw":
                    result[
                        "locality_confidence"
                    ],
            })

            print(
                f"[{i + 1}/{len(df)}] "
                f"{image_file} -> "
                f"date="
                f"{result['verbatimDate']!r} "
                f"loc="
                f"{result['verbatimLocality']!r} "
                f"date_conf="
                f"{result['date_confidence']:.2f} "
                f"loc_conf="
                f"{result['locality_confidence']:.2f}"
            )

        except Exception as e:

            print(
                f"[{i + 1}/{len(df)}] "
                f"{image_file} FAILED: {e}"
            )

            # Clear CUDA cache after an OOM/error.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            rows.append({

                "image_file":
                    image_file,

                "verbatimDate":
                    "MISSING",

                "verbatimLocality":
                    "MISSING",

                "date_confidence_raw":
                    0.0,

                "locality_confidence_raw":
                    0.0,
            })

    # ============================================================
    # SAVE
    # ============================================================

    out_df = pd.DataFrame(
        rows
    )

    out_df.to_csv(
        args.out,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"Wrote {len(out_df)} rows to:"
    )

    print(
        args.out
    )

    print(
        "=" * 70
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()