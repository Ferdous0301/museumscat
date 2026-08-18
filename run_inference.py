"""
Local VLM inference pipeline for the Danish Dung Beetle label
transcription task.

Model:
    Qwen/Qwen2.5-VL-3B-Instruct

Designed for Kaggle Tesla T4.

Key improvements:
    - Stronger locality-vs-metadata instructions
    - Stronger date-vs-metadata instructions
    - Explicit collector/collection metadata rejection
    - Multi-card handling
    - Exact transcription preservation
    - Higher visual resolution suitable for T4
    - Deterministic generation
    - Carefully selected few-shot examples
    - Conservative confidence
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
# PROMPT
# ================================================================

SYSTEM_PROMPT = r"""
You are an expert museum specimen-label transcription specialist.

Your ONLY task is to visually transcribe TWO fields from a specimen
label image:

1. verbatimDate
2. verbatimLocality

This is an EXACT TRANSCRIPTION task.

You are NOT doing:
- species identification
- geographic inference
- historical research
- metadata interpretation
- place-name prediction

============================================================
MOST IMPORTANT RULE
============================================================

ONLY output text that is visibly present on the specimen label.

NEVER guess.

NEVER complete an unclear word using your knowledge.

NEVER infer a locality from:
- collector name
- species
- museum
- collection name
- catalog number
- institution
- country
- nearby text
- handwriting style

If a target field cannot be confidently identified from visible
evidence, output:

"MISSING"

It is ALWAYS preferable to output MISSING rather than a plausible
but unsupported answer.

============================================================
FIELD 1: verbatimDate
============================================================

Find the COLLECTION DATE.

Transcribe it exactly as visible.

Preserve:
- punctuation
- spaces
- slash notation
- dots
- hyphens
- Roman numerals
- Arabic numerals
- abbreviations
- original spelling
- original ordering

Examples:

27.IV.2022
7/6 1870
1.7.2000
22.5.1977.
Juli 1930
15.V.2011

DO NOT normalize dates.

For example:

27.IV.2022

must remain:

27.IV.2022

Do NOT convert it to:

27.04.2022

============================================================
DATE DISAMBIGUATION
============================================================

A label can contain several dates.

Do NOT automatically assume that every date is the collection date.

Prefer a date that is visually associated with:
- the specimen
- collecting information
- locality
- collection event
- field information

Be cautious with dates associated with:
- determination
- identification
- accession
- cataloging
- museum processing
- later annotations

If multiple physical specimen cards contain separate collection
dates, preserve ALL relevant collection dates in their visual order,
separated by:

" | "

Example:

"5/2 53 | 22-9-36"

Do not invent additional dates.

============================================================
FIELD 2: verbatimLocality
============================================================

Find the GEOGRAPHIC PLACE where the specimen was collected.

The locality must represent a physical geographic place.

Examples:

Dyrehaven
Tisvilde
Lodskovvad
Svinø strand
Bovbj.
Ørholm
Røsnæsgd. NWZ

Preserve exactly what is visible.

Preserve:
- spelling
- capitalization
- abbreviations
- punctuation
- Danish letters such as ø, æ, å
- historical spelling

DO NOT translate.

DO NOT expand abbreviations.

============================================================
VERY IMPORTANT: LOCALITY VS METADATA
============================================================

A large amount of text on museum labels is NOT locality.

Do NOT output the following as locality merely because it appears
near the date:

- collector names
- person's names
- collection names
- museum names
- institution names
- catalog information
- accession information
- determination information
- "coll."
- "det."
- "Tilg."
- collection numbers
- specimen numbers
- taxonomic information

For example, if the label visually contains:

"Dania coll. O. Mic. Hansen"

DO NOT output that entire phrase as locality.

"Dania" is a collection/museum-related term, NOT a geographic
locality.

"O. Mic. Hansen" is a person's name, NOT a locality.

Therefore:

verbatimLocality = "MISSING"

UNLESS there is a separate visible geographic place name.

============================================================
CRITICAL LOCALITY DECISION
============================================================

Before outputting a locality, ask yourself:

"Is this text actually the name of a geographic place?"

If NO:
    output MISSING.

If YES:
    transcribe the visible geographic name exactly.

Do NOT turn a collector name into a locality.

Do NOT turn "coll." information into a locality.

Do NOT turn "Dania" into a locality.

Do NOT turn museum information into a locality.

============================================================
SHORT LOCALITY ABBREVIATIONS
============================================================

Short text CAN be a valid locality.

Examples:

Kb
Ti
Bovbj.

Do NOT reject a short locality merely because it is short.

However, the text must actually function as a geographic locality.

Use visual context to distinguish:

"Ti"

from a person's initials, collection code, or other metadata.

============================================================
HABITAT / SUBSTRATE
============================================================

Habitat and substrate are NOT locality.

Examples:

"i kogødning"
"in cow dung"

must NOT become locality.

If the label says:

Dyrehaven
i kogødning

then locality should be:

"Dyrehaven"

not:

"Dyrehaven | i kogødning"

============================================================
MULTIPLE PHYSICAL CARDS
============================================================

Inspect the ENTIRE IMAGE.

Some specimen photographs contain multiple physical cards,
labels, slips, or handwritten notes.

Do NOT stop after finding the first date or locality.

Inspect every visible card.

If several distinct collection records are present, extract ALL
relevant values.

Separate multiple values using:

" | "

Example:

verbatimDate:
"5/2 53 | 22-9-36"

Example:

verbatimLocality:
"Place A | Place B"

Do NOT merge unrelated metadata into the locality.

============================================================
HANDWRITING
============================================================

This is transcription, not correction.

If the character appears to be:

5

write:

5

If it appears to be:

S

write:

S

If a Danish character is visible, preserve it.

Do not modernize historical spelling.

Do not translate Danish place names.

Do not expand abbreviations.

If an entire field cannot be reliably read, use:

"MISSING"

However, do NOT output MISSING merely because handwriting is
slightly difficult.

If enough characters are visibly supported to produce a reasonable
transcription, transcribe them.

============================================================
VISUAL INSPECTION PROCEDURE
============================================================

Before answering, perform this internal procedure:

STEP 1:
Inspect the entire image.

STEP 2:
Locate every visible physical label/card.

STEP 3:
Read all text surrounding the specimen.

STEP 4:
Identify candidate dates.

STEP 5:
Determine which candidate is the collection date.

STEP 6:
Identify candidate place names.

STEP 7:
Reject collector names, museum names, collection names,
catalog information and other metadata as locality.

STEP 8:
Check for additional cards.

STEP 9:
Re-read difficult characters.

STEP 10:
Return ONLY the final JSON.

============================================================
IMPORTANT ANTI-HALLUCINATION RULE
============================================================

Do not use your world knowledge to fill gaps.

For example, if you see:

"Dania coll. O. Mic. Hansen"

you must NOT decide that the locality is some Danish place
because the specimen is known to come from Denmark.

The image itself must provide evidence.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Do not write:
- markdown
- explanations
- analysis
- bullet points
- comments
- code fences

Return exactly:

{
  "verbatimDate": "<string or MISSING>",
  "verbatimLocality": "<string or MISSING>",
  "date_confidence": 0.0,
  "locality_confidence": 0.0
}

============================================================
CONFIDENCE
============================================================

Confidence describes visual certainty.

0.95-1.00:
Very clear visible text.

0.80-0.94:
Mostly clear, with minor uncertainty.

0.50-0.79:
Some characters or interpretation are uncertain.

0.20-0.49:
Very difficult to read.

0.00-0.19:
Missing or unsupported.

Do NOT give high confidence simply because you produced an answer.

If locality is inferred from metadata rather than clearly visible
geographic evidence, it MUST be MISSING with low confidence.

If a field is MISSING, confidence should normally be 0.00-0.05.
"""


USER_PROMPT = r"""
Inspect the entire specimen image carefully.

Extract ONLY:

1. The collection date.
2. The geographic collection locality.

Before answering, specifically check whether apparent locality text
is actually:
- a collector name
- a person's name
- a collection name
- museum metadata
- catalog information
- determination information

If it is metadata rather than a geographic place, DO NOT output it
as locality.

Remember:

- "Dania" is NOT a locality.
- "coll." indicates collection/collector metadata.
- A person's name is NOT a locality.
- Habitat/substrate is NOT locality.
- Short abbreviations can still be valid localities.
- Preserve exact visible spelling.
- Preserve punctuation.
- Preserve Danish characters.
- Do not normalize dates.
- Inspect every physical card.
- Do not guess missing text.

Return ONLY the JSON object.
"""


FEWSHOT_INSTRUCTION = """
The following are examples from the same dataset.

Use them ONLY to learn the annotation conventions.

The answers shown for these examples are ground truth.

Do NOT copy any text from an example into the target answer unless
the same text is visibly present in the target image.

Pay particular attention to:
- exact date formatting
- exact locality spelling
- MISSING fields
- multiple cards
- distinguishing geographic locality from metadata
"""


# ================================================================
# IMAGE CHECK
# ================================================================

def check_image(path: Path):

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
    Select examples that teach the model the most useful
    annotation conventions.
    """

    train_df = train_df.copy()

    selected = []

    # ------------------------------------------------------------
    # 1. MULTI-CARD EXAMPLE
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
    # 2. DATE MISSING / LOCALITY PRESENT
    # ------------------------------------------------------------

    date_missing_loc_present = train_df[
        train_df["verbatimDate"].astype(str).eq("MISSING")
        &
        ~train_df["verbatimLocality"].astype(str).eq("MISSING")
    ]

    if len(date_missing_loc_present) > 0:

        candidate = date_missing_loc_present.sample(
            1,
            random_state=seed + 2,
        ).iloc[0]

        if candidate["image_file"] not in {
            x["image_file"] for x in selected
        }:
            selected.append(candidate)

    # ------------------------------------------------------------
    # 3. LOCALITY MISSING / DATE PRESENT
    # ------------------------------------------------------------

    loc_missing_date_present = train_df[
        train_df["verbatimLocality"].astype(str).eq("MISSING")
        &
        ~train_df["verbatimDate"].astype(str).eq("MISSING")
    ]

    if len(loc_missing_date_present) > 0:

        candidate = loc_missing_date_present.sample(
            1,
            random_state=seed + 3,
        ).iloc[0]

        if candidate["image_file"] not in {
            x["image_file"] for x in selected
        }:
            selected.append(candidate)

    # ------------------------------------------------------------
    # 4. BOTH MISSING
    # ------------------------------------------------------------

    both_missing = train_df[
        train_df["verbatimDate"].astype(str).eq("MISSING")
        &
        train_df["verbatimLocality"].astype(str).eq("MISSING")
    ]

    if len(both_missing) > 0:

        candidate = both_missing.sample(
            1,
            random_state=seed + 4,
        ).iloc[0]

        if candidate["image_file"] not in {
            x["image_file"] for x in selected
        }:
            selected.append(candidate)

    # ------------------------------------------------------------
    # 5. FILL REMAINING WITH NORMAL EXAMPLES
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
            random_state=seed + len(selected) + 20,
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
    # FEW SHOT
    # ------------------------------------------------------------

    if fewshot_examples:

        content.append({
            "type": "text",
            "text": FEWSHOT_INSTRUCTION.format(
                n=len(fewshot_examples)
            ),
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
    # TARGET
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

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

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

    if str(date).strip() == "MISSING":
        date_conf = min(
            date_conf,
            0.05,
        )

    if str(locality).strip() == "MISSING":
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

            # JSON output is short.
            max_new_tokens=180,

            # Deterministic OCR-style transcription.
            do_sample=False,

            # Small penalty against repeated text.
            repetition_penalty=1.03,
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
        default=4,
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
    #
    # IMPORTANT FOR TESLA T4:
    #
    # 1024 * 28 * 28 = 802,816 pixels
    #
    # This is a reasonable upper limit for your 14.6 GB T4.
    #
    # Do NOT jump back to 1536 on this GPU when using
    # four few-shot images, because it can cause OOM.
    #
    # ============================================================

    MIN_PIXELS = 384 * 28 * 28
    MAX_PIXELS = 1024 * 28 * 28

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
        "384 -> 1024 pixels",
    )

    # ============================================================
    # FEW SHOT
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

            # Free unused CUDA memory after a failure.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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


if __name__ == "__main__":
    main()