"""
Shared image preprocessing, used by both run_inference.py (the VLM
pipeline) and ocr_crosscheck.py (a CPU-only OCR cross-check that has no
reason to depend on torch/transformers/qwen_vl_utils).

Keeping this in its own module means ocr_crosscheck.py can import just the
resize logic without pulling in the entire VLM stack -- both for faster/
lighter environments and so a change or import error in the VLM stack can't
break the independent OCR cross-check.
"""

from pathlib import Path

from PIL import Image, ImageEnhance


# ================================================================
# PIXEL BUDGETS
# ================================================================

UPSCALE_FLOOR = 2

CONTRAST_FACTOR = 1.10
SHARPNESS_FACTOR = 1.25

# Target image (the one being transcribed).
TARGET_MIN_PIXELS = 384 * 28 * 28   # ~301k px
TARGET_MAX_PIXELS = 1024 * 28 * 28  # ~803k px

# Few-shot example images: lower budget, style/format only, not fine OCR.
FEWSHOT_MIN_PIXELS = 256 * 28 * 28  # ~201k px
FEWSHOT_MAX_PIXELS = 448 * 28 * 28  # ~351k px

# Escalated resolution for the rare case where both fields are still
# MISSING after the normal pass. Stays well under the known T4 OOM
# boundary (min=512*28*28, max=1536*28*28 failed on ~15GB).
ESCALATED_MAX_PIXELS = 1152 * 28 * 28  # ~903k px


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
    will actually use (see run_inference.py's module docstring for why
    this replaced the old "upscale 4x, then let the processor downsample
    again" approach -- double-resampling was blurring fine handwriting).
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