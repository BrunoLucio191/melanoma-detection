"""Image loading and preparation for lesion segmentation."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def load_image(path: str | Path) -> Image:
    """Load an image in OpenCV BGR format.

    Raises:
        ValueError: If the path does not contain a readable image.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def normalize_lighting(image: Image) -> Image:
    """Improve local contrast through CLAHE on the LAB lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    improved = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((improved, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def remove_hair(image: Image) -> Image:
    """Reduce dark hair using a black-hat mask and Telea inpainting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)


def resize_image(image: Image, max_dimension: int | None = 1024) -> Image:
    """Resize an image while keeping its aspect ratio."""
    if max_dimension is None or max(image.shape[:2]) <= max_dimension:
        return image

    height, width = image.shape[:2]
    scale = max_dimension / max(height, width)
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_image(
    image: Image, max_dimension: int | None = 1024
) -> tuple[Image, NDArray[np.float32], Image]:
    """Return the LAB image, K-means pixels, and cleaned BGR image.

    All returned images have the same dimensions. This prevents a resized mask
    from being applied to an image of a different size during feature extraction.
    """
    cleaned = remove_hair(normalize_lighting(image))
    cleaned = resize_image(cleaned, max_dimension)
    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    if np.mean(lab[:, :, 1]) > 148:
        pixels = pixels.copy()
        pixels[:, 1] = 128 + (pixels[:, 1] - 128) * 0.4

    return lab, pixels, cleaned
