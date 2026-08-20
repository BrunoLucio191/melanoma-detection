import cv2
import numpy as np
import pytest

from melanoma_detection.config import FEATURE_NAMES
from melanoma_detection.features import extract_features
from melanoma_detection.preprocessing import preprocess_image


def test_preprocessing_keeps_output_dimensions_aligned():
    image = np.full((1200, 800, 3), 160, dtype=np.uint8)

    lab, pixels, cleaned = preprocess_image(image, max_dimension=300)

    assert lab.shape == cleaned.shape
    assert pixels.shape == (lab.shape[0] * lab.shape[1], 3)


def test_feature_extractor_returns_expected_number_of_values():
    image = np.full((128, 128, 3), (100, 130, 160), dtype=np.uint8)
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(mask, (64, 64), 30, 255, -1)

    values = extract_features(image, mask)

    assert values is not None
    assert len(values) == len(FEATURE_NAMES) == 46
    assert np.isfinite(values).all()


def test_feature_extractor_rejects_mismatched_dimensions():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(ValueError, match="same dimensions"):
        extract_features(image, mask)
