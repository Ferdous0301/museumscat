"""
Local VLM inference pipeline for the Danish Dung Beetle label
transcription task.

Model:
    Qwen/Qwen2.5-VL-3B-Instruct

Designed for Kaggle Tesla T4 GPU.

This version:
    - Uses a stronger transcription prompt
    - Uses higher visual resolution
    - Uses deterministic generation
    - Uses carefully selected few-shot examples
    - Strongly discourages hallucination
    - Preserves exact visible text
    - Handles MISSING and multi-card specimens
    - Produces the same CSV format as the previous pipeline
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
You are an expert museum archivist performing EXACT visual transcription
of specimen labels from the Natural History Museum of Denmark.

Your task is NOT to identify the species and NOT to infer information.

You must extract exactly two target fields from the visible specimen label:

1. verbatimDate
   The collection date exactly as written.

2. verbatimLocality
   The collection locality exactly as written.

============================================================
CRITICAL RULE: TRANSCRIBE, DO NOT GUESS
============================================================

Only report text that is visibly present in the image.

DO NOT invent text.
DO NOT infer a date from context.
DO NOT infer a locality from a species, collector, museum or catalog.
DO NOT "correct" unclear handwriting into a plausible word.

If you cannot find convincing visible evidence for a target field,
output:

"MISSING"

It is much better to output MISSING than to invent a date or locality.

============================================================
DATE RULES
============================================================

Report the collection date exactly as written.

Preserve:
- punctuation
- spaces
- Roman numerals
- abbreviations
- historical spelling
- the original ordering

Examples:

27.IV.2022
7/6 1870
1.7.2000
22.5.1977.
Juli 1930

Do NOT convert:

27.IV.2022

into:

27.04.2022

Do NOT replace Roman numerals with Arabic month numbers.

Be especially careful not to confuse:
- collection dates
- determination dates
- accession/catalog dates
- museum metadata dates
- dates associated with collector changes

If multiple physical specimen cards/labels contain distinct collection
information, report ALL relevant dates separated by:

" | "

============================================================
LOCALITY RULES
============================================================

Report the geographic collection locality exactly as visibly written.

Preserve:
- spelling
- abbreviations
- capitalization
- Danish characters such as ø, æ and å
- punctuation
- locality hierarchy

Examples of possible locality information include:

Dyrehaven
Tisvilde
Lodskovvad
Svinø strand
Kb
Bovbj.

Short locality abbreviations MUST NOT automatically be discarded.

If a short label such as "Ti" or "Kb" is visibly used as the
collection locality, transcribe it exactly.

However, ordinary museum/institution metadata is NOT automatically a
locality.

============================================================
MUSEUM / COLLECTION METADATA
============================================================

"Dania" is NEVER a locality.

It is the name of the collection.

If the label contains only "Dania" as a possible locality,
output:

"MISSING"

Do not treat museum or collection catalog information as geographic
locality merely because it is printed near a date.

Examples of metadata that may NOT be collection locality include:

- collector information
- "Coll."
- determination information
- "det."
- "Tilg."
- museum/catalog information
- collection names

BUT:

Do not blindly remove short text.

If the visible label clearly uses a short abbreviation as the locality,
transcribe that abbreviation.

============================================================
HABITAT / SUBSTRATE
============================================================

Habitat or substrate descriptions are NOT locality.

For example:

"i kogødning"

means "in cow dung" and should NOT become the locality.

Keep the geographic place name, but remove habitat/substrate text from
the locality field.

============================================================
MULTI-CARD SPECIMENS
============================================================

Inspect the ENTIRE IMAGE before answering.

Some specimens have multiple physical labels/cards around the pin.

If multiple distinct collection records are visible, extract ALL of them.

Separate multiple values with:

" | "

For example:

verbatimDate:
"5/2 53 | 22-9-36"

or:

verbatimLocality:
"Place A | Place B"

Do this even if values on separate cards appear identical.

============================================================
HANDWRITING
============================================================

This is a transcription task.

Do not silently normalize handwriting.

If a character is clearly visible as "5", write "5".

If it is clearly "S", write "S".

If a Danish character such as "ø" is visible, preserve "ø".

Do not translate Danish place names.

Do not expand abbreviations.

Do not replace historical spelling with modern spelling.

When a character is genuinely unreadable, prefer MISSING for the
field rather than inventing a complete word.

============================================================
VISUAL INSPECTION PROCEDURE
============================================================

Before producing the answer:

1. Inspect the whole specimen image.
2. Locate every physical label/card.
3. Identify text that appears to represent collection information.
4. Separate date information from locality information.
5. Ignore collector/determination/museum metadata.
6. Check for additional cards.
7. Re-read the characters carefully.
8. Only then produce the JSON.

============================================================
OUTPUT
============================================================

Respond with ONLY one valid JSON object.

No markdown.
No explanation before the JSON.
No explanation after the JSON.

Required format:

{
  "verbatimDate": "<string or MISSING>",
  "verbatimLocality": "<string or MISSING>",
  "date_confidence": 0.0,
  "locality_confidence": 0.0,
  "reasoning": "<one short sentence>"
}

============================================================
CONFIDENCE
============================================================

Confidence must reflect the visual evidence.

0.95-1.00:
Very clear text and very little ambiguity.

0.80-0.94:
Mostly clear but minor character uncertainty.

0.50-0.79:
Some uncertainty or difficult handwriting.

0.20-0.49:
Significant uncertainty.

0.00-0.19:
The field is missing or the model cannot reliably read it.

IMPORTANT:

Do NOT give 0.90 or 1.00 merely because you produced an answer.

A visibly uncertain transcription must receive lower confidence.

A hallucinated value should have LOW confidence.

If a field is MISSING, its confidence should normally be near 0.0.
"""


USER_PROMPT = r"""
Carefully inspect this specimen image and transcribe the collection
date and collection locality.

Remember:

- Look at the entire image.
- Look for multiple physical labels/cards.
- Transcribe visible text exactly.
- Do not invent missing information.
- Do not treat museum metadata as locality.
- "Dania" is never locality.
- Preserve Danish characters and abbreviations.
- Use "MISSING" when the target field genuinely cannot be identified.

Return ONLY the required JSON object.
"""


FEWSHOT_INSTRUCTION = """
Here are {n} examples from the same dataset.

Use them ONLY to understand the annotation conventions.

The example answers are ground truth.

Do not copy their text into the new answer unless that text is actually
visible on the target image.
"""


# ================================================================
# IMAGE CHECK
# ================================================================

def check_image(path: Path):
    """Verify that an image exists and is readable."""

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
    Select useful examples rather than completely random examples.

    Priority:
        1. multi-card examples
        2. MISSING examples
        3. ordinary examples
    """

    train_df = train_df.copy()

    selected = []

    # ------------------------------------------------------------
    # MULTI-CARD
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
    # MISSING
    # ------------------------------------------------------------

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
    # FEW-SHOT
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
    # MISSING
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

    # Missing should not have high confidence
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
    # PROCESS IMAGE
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
    # GENERATE
    # ------------------------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,

            # JSON is short. 150 tokens is plenty.
            max_new_tokens=180,

            # Deterministic transcription
            do_sample=False,

            # Avoid unnecessary repetition
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
    # HIGHER VISUAL RESOLUTION
    # ============================================================

    processor = AutoProcessor.from_pretrained(

        MODEL_NAME,

        # Minimum visual resolution
        min_pixels=512 * 28 * 28,

        # Increased maximum resolution.
        #
        # Your previous value was:
        # 1280 * 28 * 28
        #
        # We increase it moderately for tiny text.
        max_pixels=1536 * 28 * 28,
    )

    print(
        "Model loaded successfully."
    )

    print(
        "Visual resolution:",
        "min_pixels=512*28*28",
        "max_pixels=1536*28*28",
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