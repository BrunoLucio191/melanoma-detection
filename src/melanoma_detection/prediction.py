"""Prediction workflow for one dermoscopic image."""

import warnings
from pathlib import Path

import cv2
import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from .config import DEFAULT_MODEL_PATH
from .features import extract_features
from .preprocessing import load_image, preprocess_image
from .segmentation import segment_lesion


def _load_model(model_path: Path) -> tuple[dict, bool]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", InconsistentVersionWarning)
        package = joblib.load(model_path)
    version_mismatch = any(
        isinstance(item.message, InconsistentVersionWarning) for item in caught_warnings
    )
    required_keys = {"model", "scaler", "threshold", "features"}
    if not isinstance(package, dict) or not required_keys.issubset(package):
        raise ValueError("The model package has an unsupported format")
    return package, version_mismatch


def _predict_with_package(image_path: Path, package: dict) -> float:
    feature_names = list(package["features"])

    image = load_image(image_path)
    lab_image, pixels, cleaned_image = preprocess_image(image)
    mask = segment_lesion(lab_image, pixels)
    values = extract_features(cleaned_image, mask)
    if values is None:
        raise RuntimeError("No lesion was found in the segmented mask")
    if len(values) != len(feature_names):
        raise ValueError("Model and feature extractor are not compatible")

    feature_table = pd.DataFrame([values], columns=feature_names)
    scaled = package["scaler"].transform(feature_table)
    return float(package["model"].predict_proba(scaled)[0, 1])


def predict_image(image_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> float:
    """Return the model's melanoma-class probability for one image."""
    package, _ = _load_model(model_path)
    return _predict_with_package(image_path, package)


def show_prediction(
    image_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    show_window: bool = False,
) -> None:
    """Print a prediction and optionally display the segmented lesion boundary."""
    package, version_mismatch = _load_model(model_path)
    probability = _predict_with_package(image_path, package)
    threshold = float(package["threshold"])
    predicted_class = "melanoma" if probability >= threshold else "nevus"

    print(f"Predicted class: {predicted_class}")
    print(f"Melanoma-class probability: {probability:.1%}")
    print(f"Decision threshold: {threshold:.2f}")
    if version_mismatch:
        print("Warning: this model was created with a different scikit-learn version.")
        print("Retrain it before using its results in an evaluation.")
    print("This result is experimental and is not a medical diagnosis.")

    if not show_window:
        return

    image = load_image(image_path)
    lab_image, pixels, cleaned_image = preprocess_image(image)
    mask = segment_lesion(lab_image, pixels)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = cleaned_image.copy()
    color = (0, 0, 255) if probability >= threshold else (0, 180, 0)
    cv2.drawContours(display, contours, -1, color, 2)
    cv2.imshow("Lesion segmentation - press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
