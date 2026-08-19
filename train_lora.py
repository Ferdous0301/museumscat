"""
LoRA fine-tuning for Qwen2.5-VL-3B-Instruct on your labeled specimen-label
training data.

WHY THIS EXISTS: few-shot prompting teaches the model annotation
CONVENTIONS (formatting, MISSING usage, locality-vs-metadata distinction)
fresh on every single inference call, using only 2-3 examples' worth of
context. Fine-tuning bakes what your ~200 labeled rows collectively teach
about THIS dataset -- its handwriting styles, card layouts, common
localities, common metadata phrasing -- directly into the model's weights.
It does NOT, and cannot, fix cases where the text is genuinely illegible at
the resolution used (the "genuine resolution ceiling" cases found during
testing) -- that's a pixels problem, not a "does the model understand the
task" problem.

DESIGN DECISIONS (T4-specific):
  - fp16, not bf16. T4 is Turing architecture and lacks native bf16 tensor
    core support -- bf16 "works" via emulation but is slower and buys
    nothing on this GPU. Mixed-precision training here uses fp16 +
    GradScaler for numerical stability.
  - Vision encoder frozen entirely. LoRA is applied only to the language-
    model attention projections. This is both a VRAM saving (no optimizer
    state or gradients for the (larger) vision tower) and a reasonable
    prior: the vision tower's job (extract visual features) is fairly
    task-agnostic, while what actually needs adapting to this dataset is
    how the language side interprets those features and formats output.
  - No batching across images. Qwen2.5-VL's dynamic resolution means
    different images produce different numbers of visual tokens, which
    makes naive batching either wasteful (padding to the longest) or
    complex (custom collation). With only ~200 training rows and a T4's
    limited VRAM, a per-example loop with gradient accumulation for an
    effective batch size is simpler and avoids that complexity entirely.
  - No few-shot examples in the training prompt. The model is being
    taught the task directly through weight updates on each single-image
    example; baking in a few-shot preamble would waste both training
    compute and context length on a mechanism that's no longer needed for
    a fine-tuned model to that degree (though nothing stops you from
    still using --n-fewshot 1-2 at inference time on top of the adapter
    for extra steering).
  - Held-out validation split with best-checkpoint saving by val loss,
    since ~200 rows is small enough that overfitting within a handful of
    epochs is a real risk, not a hypothetical one.

INSTALL (Kaggle):
    !pip install -q peft --break-system-packages

USAGE:
    python train_lora.py \
        --train-csv train.csv \
        --images /kaggle/input/.../images \
        --output-dir /kaggle/working/lora_adapter \
        --epochs 3 --lr 1e-4 --grad-accum-steps 8

Then point run_inference.py at the result:
    python run_inference.py ... --lora-adapter /kaggle/working/lora_adapter/best
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)

from qwen_vl_utils import process_vision_info

from prompt_template import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from image_utils import smart_resize_image, TARGET_MIN_PIXELS, TARGET_MAX_PIXELS

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise SystemExit(
        "peft is required for LoRA fine-tuning. Run:\n"
        "  pip install peft --break-system-packages"
    )


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# ================================================================
# TRAINING EXAMPLE CONSTRUCTION
# ================================================================

def target_json_for_row(row: pd.Series) -> str:
    """
    Synthesize the target completion for one training row. Confidence
    targets are binary (1.0 if the field is present, 0.0 if MISSING) --
    we don't have genuine per-field confidence ground truth, only
    presence/absence, so teaching a nuanced confidence scale isn't
    supported by the data. The model's raw output confidence remains a
    fairly weak signal even after fine-tuning for this reason; that's
    exactly why risk_ranking.py exists as a separate, better-calibrated
    layer on top rather than relying on this number directly.
    """
    date = str(row["verbatimDate"])
    locality = str(row["verbatimLocality"])
    answer = {
        "verbatimDate": date,
        "verbatimLocality": locality,
        "date_confidence": 0.0 if date.strip().upper() == "MISSING" else 1.0,
        "locality_confidence": 0.0 if locality.strip().upper() == "MISSING" else 1.0,
    }
    return json.dumps(answer, ensure_ascii=False)


def build_training_messages(image_path: Path, target_text: str):
    """
    Builds the full (user + assistant) conversation for one training
    example. No few-shot examples -- see module docstring.
    """
    user_content = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": USER_PROMPT_TEMPLATE},
        {"type": "image", "image": smart_resize_image(
            image_path, min_pixels=TARGET_MIN_PIXELS, max_pixels=TARGET_MAX_PIXELS
        )},
    ]

    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    prompt_only_messages = [{"role": "user", "content": user_content}]

    return full_messages, prompt_only_messages


def build_example_tensors(processor, image_path: Path, target_text: str, device):
    """
    Returns (inputs dict, labels tensor) for one training example, with
    the prompt portion of labels masked to -100 (ignored by the loss) so
    the model is only supervised on generating the assistant's JSON
    completion, not on reproducing the (very long) system prompt.
    """
    full_messages, prompt_only_messages = build_training_messages(image_path, target_text)

    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        prompt_only_messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(full_messages)

    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        padding=False, return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=False, return_tensors="pt",
    )

    prompt_len = prompt_inputs.input_ids.shape[1]
    full_len = full_inputs.input_ids.shape[1]

    labels = full_inputs.input_ids.clone()
    if prompt_len >= full_len:
        # Defensive: if something about chat-template formatting made the
        # "prompt-only" rendering as long as (or longer than) the full
        # rendering, there's nothing safe left to supervise -- skip this
        # example rather than risk masking everything or nothing.
        return None, None
    labels[:, :prompt_len] = -100

    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
    labels = labels.to(device)

    return full_inputs, labels


# ================================================================
# MODEL SETUP
# ================================================================

def build_model(lora_r: int, lora_alpha: int, lora_dropout: float, target_mlp: bool):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Freeze everything first, then let LoRA injection re-enable grad only
    # on its own adapter matrices. Freezing explicitly (rather than
    # relying on peft's default behavior) also protects against LoRA's
    # target_modules substring-matching accidentally hitting a
    # similarly-named layer inside the vision tower.
    for p in model.parameters():
        p.requires_grad = False

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Explicitly re-freeze the vision tower in case target_modules matched
    # anything under it by accident (defensive, cheap to check).
    n_unfrozen_vision = 0
    for name, p in model.named_parameters():
        if "visual" in name and p.requires_grad:
            p.requires_grad = False
            n_unfrozen_vision += 1
    if n_unfrozen_vision:
        print(f"WARNING: re-froze {n_unfrozen_vision} unexpectedly-unfrozen "
              f"vision-tower parameters (LoRA target_modules matched inside "
              f"the vision encoder). If this number is large, check "
              f"target_modules against your installed Qwen2.5-VL version's "
              f"actual module names.")

    model.print_trainable_parameters()

    # Required when combining gradient checkpointing with a (mostly)
    # frozen base model + LoRA: without this, gradients don't propagate
    # correctly back through the frozen layers into the LoRA adapters
    # during checkpointed recomputation. This is a well-known peft +
    # gradient-checkpointing interaction, not optional.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required alongside gradient checkpointing

    return model, device


# ================================================================
# TRAIN / VAL SPLIT
# ================================================================

def train_val_split(df: pd.DataFrame, val_fraction: float, seed: int):
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val_idx = set(idx[:n_val])
    train_idx = [i for i in idx if i not in val_idx]
    return df.loc[train_idx].reset_index(drop=True), df.loc[list(val_idx)].reset_index(drop=True)


# ================================================================
# EVAL
# ================================================================

@torch.no_grad()
def evaluate(model, processor, val_df, images_dir, device):
    model.eval()
    losses = []
    for _, row in val_df.iterrows():
        image_path = images_dir / str(row["image_file"])
        if not image_path.exists():
            continue
        target_text = target_json_for_row(row)
        inputs, labels = build_example_tensors(processor, image_path, target_text, device)
        if inputs is None:
            continue
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
            out = model(**inputs, labels=labels)
        losses.append(out.loss.item())
    model.train()
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


# ================================================================
# MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum-steps", type=int, default=8,
                     help="Effective batch size = grad_accum_steps, since each "
                          "example is processed individually (see module docstring).")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-mlp", action="store_true",
                     help="Also apply LoRA to MLP layers (gate/up/down_proj), not just "
                          "attention. Higher quality ceiling, more VRAM and training time.")
    ap.add_argument("--eval-every-steps", type=int, default=20,
                     help="Run validation every N optimizer steps (not N examples).")
    ap.add_argument("--max-steps", type=int, default=None,
                     help="Optional hard cap on optimizer steps, for a quick smoke test "
                          "before committing to a full run.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    df = pd.read_csv(args.train_csv)
    train_df, val_df = train_val_split(df, args.val_fraction, args.seed)
    print(f"Train rows: {len(train_df)}  Val rows: {len(val_df)}")

    print(f"Loading {MODEL_NAME} for LoRA fine-tuning...")
    model, device = build_model(args.lora_r, args.lora_alpha, args.lora_dropout, args.target_mlp)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        min_pixels=TARGET_MIN_PIXELS,
        max_pixels=TARGET_MAX_PIXELS,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    steps_per_epoch = max(1, len(train_df) // args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    print(f"Total optimizer steps planned: {total_steps} "
          f"({steps_per_epoch}/epoch x {args.epochs} epochs, "
          f"capped at --max-steps={args.max_steps})")

    model.train()

    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    best_val_loss = float("inf")

    stop = False
    for epoch in range(args.epochs):
        if stop:
            break

        epoch_df = train_df.sample(frac=1.0, random_state=args.seed + epoch).reset_index(drop=True)

        for i, row in epoch_df.iterrows():
            image_path = images_dir / str(row["image_file"])
            if not image_path.exists():
                print(f"  [skip] image not found: {image_path}")
                continue

            target_text = target_json_for_row(row)

            try:
                inputs, labels = build_example_tensors(processor, image_path, target_text, device)
            except Exception as e:
                print(f"  [skip] failed to build example for {row['image_file']}: {e}")
                continue

            if inputs is None:
                print(f"  [skip] degenerate prompt/completion split for {row['image_file']}")
                continue

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16):
                out = model(**inputs, labels=labels)
                loss = out.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += out.loss.item()
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = accum_loss / args.grad_accum_steps
                print(f"epoch {epoch} step {global_step}/{total_steps} "
                      f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

                if global_step % args.eval_every_steps == 0:
                    val_loss = evaluate(model, processor, val_df, images_dir, device)
                    print(f"  [eval] step {global_step} val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = output_dir / "best"
                        model.save_pretrained(best_dir)
                        print(f"  [checkpoint] new best val_loss={val_loss:.4f}, saved to {best_dir}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"Reached --max-steps={args.max_steps}, stopping.")
                    stop = True
                    break

    final_val_loss = evaluate(model, processor, val_df, images_dir, device)
    print(f"Final val_loss={final_val_loss:.4f} (best seen: {best_val_loss:.4f})")

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")

    if best_val_loss < float("inf"):
        print(f"Best adapter (by val loss) is at {output_dir / 'best'} -- "
              f"use that with run_inference.py --lora-adapter, not 'final', "
              f"unless final's val_loss is actually the lowest you saw above.")
    else:
        print("No checkpoint improved on the initial best_val_loss threshold "
              "(this can happen with very few eval points, e.g. --max-steps set "
              "low for a smoke test) -- use 'final' in that case.")


if __name__ == "__main__":
    main()