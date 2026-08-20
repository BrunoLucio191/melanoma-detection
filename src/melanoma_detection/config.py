"""Shared paths and feature names used by the pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "HAM10000_organizado"
DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset_melanoma_raw.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "modelo_melanoma_rf_otimizado.pkl"
DEFAULT_IMPORTANCE_PATH = PROJECT_ROOT / "importancia_features.png"

GEOMETRY_FEATURES = [
    "Area",
    "Perimeter",
    "Circularity",
    "Solidity",
    "AspectRatio",
    "Extent",
    "BorderIrregularity",
]
HU_FEATURES = [f"Hu_{index}" for index in range(1, 8)]
HSV_FEATURES = [
    f"{channel}_{statistic}"
    for channel in ("H", "S", "V")
    for statistic in ("mean", "std", "skew", "kurt")
]
LAB_FEATURES = [
    f"{channel}_{statistic}"
    for channel in ("L", "a", "b")
    for statistic in ("mean", "std", "skew", "kurt")
]
TEXTURE_FEATURES = [
    "Contrast",
    "Dissimilarity",
    "Homogeneity",
    "Energy",
    "Correlation",
    "Entropy",
]
ASYMMETRY_FEATURES = ["Asymmetry_H", "Asymmetry_V"]

FEATURE_NAMES = (
    GEOMETRY_FEATURES
    + HU_FEATURES
    + HSV_FEATURES
    + LAB_FEATURES
    + TEXTURE_FEATURES
    + ASYMMETRY_FEATURES
)
