"""Dataset processing and feature table generation."""

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from .config import DEFAULT_CSV_PATH, DEFAULT_DATA_DIR, FEATURE_NAMES
from .features import extract_features
from .preprocessing import load_image, preprocess_image
from .segmentation import segment_lesion

CLASS_DIRECTORIES = {"nevus_melanocitico": 0, "melanoma": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _find_images(data_dir: Path) -> list[tuple[Path, int]]:
    images: list[tuple[Path, int]] = []
    for directory, label in CLASS_DIRECTORIES.items():
        class_dir = data_dir / directory
        if not class_dir.exists():
            continue
        images.extend(
            (path, label)
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return images


def _process_image(
    image_path: Path, label: int, mask_dir: Path, features_only: bool
) -> list[float | int | str] | None:
    class_name = "melanoma" if label == 1 else "nevus"
    mask_path = mask_dir / class_name / f"{image_path.stem}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    )

    image = load_image(image_path)
    lab_image, pixels, cleaned_image = preprocess_image(image)
    if mask is None or mask.shape != cleaned_image.shape[:2]:
        mask = segment_lesion(lab_image, pixels)
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Could not save mask: {mask_path}")

    if not features_only:
        return []
    values = extract_features(cleaned_image, mask)
    if values is None:
        return None
    return [*values, label, image_path.name]


def build_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    masks_only: bool = False,
) -> None:
    """Create cached masks and optionally write the feature CSV file."""
    data_dir = data_dir.resolve()
    images = _find_images(data_dir)
    if not images:
        expected = ", ".join(CLASS_DIRECTORIES)
        raise FileNotFoundError(
            f"No images found in {data_dir}. Expected class folders: {expected}"
        )

    mask_dir = data_dir / "segmentado"
    failures: list[str] = []
    rows: list[list[float | int | str]] = []
    action = "Creating masks" if masks_only else "Extracting features"

    for image_path, label in tqdm(images, desc=action, unit="image"):
        try:
            result = _process_image(image_path, label, mask_dir, not masks_only)
            if result:
                rows.append(result)
            elif result is None:
                failures.append(image_path.name)
        except (OSError, ValueError, cv2.error) as error:
            failures.append(f"{image_path.name}: {error}")

    print(f"Processed {len(images) - len(failures)} of {len(images)} images.")
    if failures:
        print(f"Failed images: {len(failures)}")
    if masks_only:
        print(f"Masks saved in {mask_dir}")
        return
    if not rows:
        raise RuntimeError("No feature rows were generated")

    table = pd.DataFrame(rows, columns=[*FEATURE_NAMES, "Label", "Filename"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    counts = table["Label"].value_counts()
    print(f"Feature table saved to {csv_path}")
    print(f"Nevus samples: {counts.get(0, 0)}")
    print(f"Melanoma samples: {counts.get(1, 0)}")
