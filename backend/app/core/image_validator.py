"""
Image Validator - Quality checks for skin lesion images.
"""

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import structlog

logger = structlog.get_logger(__name__)

MIN_DIMENSION = 100
MAX_DIMENSION = 4096
BLUR_THRESHOLD = 100.0
MIN_FILE_SIZE = 1024
MAX_FILE_SIZE = 10 * 1024 * 1024


@dataclass
class ImageValidationResult:
    is_valid: bool
    image: Optional[Image.Image] = None
    width: int = 0
    height: int = 0
    format: Optional[str] = None
    blur_score: float = 0.0
    errors: list[str] = None
    warnings: list[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


def _calculate_blur_score(image: Image.Image) -> float:
    gray = image.convert("L")
    img_array = np.array(gray, dtype=np.float64)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(img_array, 1, mode="edge")
    h, w = img_array.shape
    result = np.zeros_like(img_array)
    for i in range(h):
        for j in range(w):
            window = padded[i : i + 3, j : j + 3]
            result[i, j] = np.sum(window * laplacian)
    return float(np.var(result))


def decode_base64_image(base64_string: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        if len(image_data) < MIN_FILE_SIZE:
            return None, f"Image too small ({len(image_data)} bytes)"
        if len(image_data) > MAX_FILE_SIZE:
            return None, f"Image too large ({len(image_data) // 1024 // 1024}MB, max 10MB)"
        image = Image.open(io.BytesIO(image_data))
        return image, None
    except base64.binascii.Error as e:
        return None, f"Invalid base64 encoding: {e}"
    except Exception as e:
        return None, f"Failed to decode image: {e}"


def validate_image(base64_string: str) -> ImageValidationResult:
    errors = []
    warnings = []
    image, decode_error = decode_base64_image(base64_string)
    if decode_error:
        return ImageValidationResult(is_valid=False, errors=[decode_error])
    img_format = image.format
    if img_format not in ("JPEG", "PNG", "WEBP", "MPO"):
        errors.append(f"Unsupported format: {img_format}. Use JPEG, PNG, or WebP.")
    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        errors.append(f"Image too small: {width}x{height}. Minimum {MIN_DIMENSION}x{MIN_DIMENSION}.")
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        warnings.append(f"Large image ({width}x{height}) may be slow to process.")
    blur_score = 0.0
    if not errors:
        blur_score = _calculate_blur_score(image)
        if blur_score < BLUR_THRESHOLD:
            warnings.append(f"Image may be blurry (score: {blur_score:.0f}). Results may be less reliable.")
    is_valid = len(errors) == 0
    if is_valid:
        logger.info("image_validated", width=width, height=height, format=img_format, blur_score=round(blur_score, 1))
    else:
        logger.warning("image_validation_failed", errors=errors)
    return ImageValidationResult(
        is_valid=is_valid,
        image=image if is_valid else None,
        width=width,
        height=height,
        format=img_format,
        blur_score=blur_score,
        errors=errors,
        warnings=warnings,
    )
