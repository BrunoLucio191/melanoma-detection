"""Extraction of shape, color, texture, and asymmetry features."""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from .config import FEATURE_NAMES
from .preprocessing import Image

Mask = NDArray[np.uint8]


def _geometry(contour: NDArray[np.int32]) -> list[float]:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    _, _, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / height if height > 0 else 0.0
    rectangle_area = width * height
    extent = area / rectangle_area if rectangle_area > 0 else 0.0
    border_irregularity = perimeter**2 / area if area > 0 else 0.0

    return [
        area,
        perimeter,
        circularity,
        solidity,
        aspect_ratio,
        extent,
        border_irregularity,
    ]


def _hu_moments(mask: Mask) -> list[float]:
    moments = cv2.HuMoments(cv2.moments(mask)).flatten()
    return [
        float(-np.sign(value) * np.log10(abs(value))) if value != 0 else 0.0
        for value in moments
    ]


def _color_statistics(image: Image, mask: Mask, color_space: str) -> list[float]:
    conversion = {
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2LAB,
    }[color_space]
    converted = cv2.cvtColor(image, conversion)
    selected = mask > 0
    values: list[float] = []

    for channel in cv2.split(converted):
        pixels = channel[selected]
        if pixels.size == 0:
            values.extend([0.0] * 4)
            continue
        standard_deviation = float(np.std(pixels))
        values.extend(
            [
                float(np.mean(pixels)),
                standard_deviation,
                float(skew(pixels)) if standard_deviation > 0 else 0.0,
                float(kurtosis(pixels)) if standard_deviation > 0 else 0.0,
            ]
        )
    return values


def _texture(image: Image, mask: Mask) -> list[float]:
    points = cv2.findNonZero(mask)
    if points is None:
        return [0.0] * 6

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, width, height = cv2.boundingRect(points)
    region = gray[y : y + height, x : x + width]
    region_mask = mask[y : y + height, x : x + width]
    region = cv2.bitwise_and(region, region, mask=region_mask)

    matrix = graycomatrix(
        region,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2],
        levels=256,
        symmetric=True,
        normed=True,
    )
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
    ]
    values = [float(np.mean(graycoprops(matrix, name))) for name in properties]
    values.append(float(shannon_entropy(region)))
    return values


def _asymmetry(mask: Mask) -> list[float]:
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return [0.0, 0.0]

    center_x = round(moments["m10"] / moments["m00"])
    center_y = round(moments["m01"] / moments["m00"])
    height, width = mask.shape
    transform = np.float32(
        [[1, 0, width // 2 - center_x], [0, 1, height // 2 - center_y]]
    )
    centered = cv2.warpAffine(mask, transform, (width, height))
    lesion_area = cv2.countNonZero(centered)
    if lesion_area == 0:
        return [0.0, 0.0]

    horizontal = cv2.bitwise_xor(centered, cv2.flip(centered, 1))
    vertical = cv2.bitwise_xor(centered, cv2.flip(centered, 0))
    return [
        cv2.countNonZero(horizontal) / lesion_area,
        cv2.countNonZero(vertical) / lesion_area,
    ]


def extract_features(image: Image, mask: Mask) -> list[float] | None:
    """Extract the 46 values expected by the classifier."""
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must have the same dimensions")

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    values = (
        _geometry(contour)
        + _hu_moments(binary_mask)
        + _color_statistics(image, binary_mask, "hsv")
        + _color_statistics(image, binary_mask, "lab")
        + _texture(image, binary_mask)
        + _asymmetry(binary_mask)
    )
    clean_values = [float(np.nan_to_num(value)) for value in values]
    if len(clean_values) != len(FEATURE_NAMES):
        raise RuntimeError("Unexpected number of extracted features")
    return clean_values
