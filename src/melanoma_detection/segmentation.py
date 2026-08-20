"""K-means and GrabCut segmentation for dermoscopic images."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .features import Mask
from .preprocessing import Image


def _useful_image_area(lab_image: Image) -> tuple[Mask, bool]:
    """Return a mask that excludes a dark circular dermoscope border."""
    height, width = lab_image.shape[:2]
    center = (width // 2, height // 2)
    maximum_radius = min(height, width) // 2
    lightness = lab_image[:, :, 0]

    row = lightness[center[1], :]
    column = lightness[:, center[0]]
    visible_x = np.flatnonzero(row > 15)
    visible_y = np.flatnonzero(column > 15)
    horizontal_radius = (
        (int(visible_x[-1]) - int(visible_x[0])) // 2
        if visible_x.size
        else maximum_radius
    )
    vertical_radius = (
        (int(visible_y[-1]) - int(visible_y[0])) // 2
        if visible_y.size
        else maximum_radius
    )
    detected_radius = min(horizontal_radius, vertical_radius)

    border_width = max(1, min(height, width) // 15)
    border_mean = np.mean(
        [
            lightness[:, :border_width].mean(),
            lightness[:, -border_width:].mean(),
            lightness[:border_width, :].mean(),
            lightness[-border_width:, :].mean(),
        ]
    )
    center_mean = lightness[
        height // 4 : 3 * height // 4, width // 4 : 3 * width // 4
    ].mean()
    brightness_difference = center_mean - border_mean

    has_vignette = detected_radius < maximum_radius * 0.9 or brightness_difference > 35
    if not has_vignette:
        return np.full((height, width), 255, dtype=np.uint8), False

    mask = np.zeros((height, width), dtype=np.uint8)
    safe_radius = max(10, detected_radius - 20)
    cv2.circle(mask, center, safe_radius, 255, -1)
    return mask, True


def _assign_all_pixels(
    pixels: NDArray[np.float32], centers: NDArray[np.float32]
) -> NDArray[np.int64]:
    """Assign pixels to the closest K-means center in memory-safe batches."""
    labels = np.empty(pixels.shape[0], dtype=np.int64)
    batch_size = 100_000
    for start in range(0, pixels.shape[0], batch_size):
        batch = pixels[start : start + batch_size]
        distances = np.sum((batch[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start : start + batch_size] = np.argmin(distances, axis=1)
    return labels


def _run_kmeans(
    pixels: NDArray[np.float32], height: int, width: int, clusters: int = 4
) -> NDArray[np.int64]:
    sample_limit = 500_000
    if len(pixels) > sample_limit:
        generator = np.random.default_rng(42)
        indexes = generator.choice(len(pixels), sample_limit, replace=False)
        training_pixels = pixels[indexes]
    else:
        training_pixels = pixels

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    attempts = 3 if height * width < sample_limit else 5
    _, sample_labels, centers = cv2.kmeans(
        training_pixels,
        clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    if training_pixels is pixels:
        return sample_labels.flatten()
    return _assign_all_pixels(pixels, centers.reshape(clusters, 3))


def _select_lesion_clusters(
    lab_image: Image,
    labels: NDArray[np.int64],
    useful_area: Mask,
    has_vignette: bool,
) -> tuple[Mask, list[int]]:
    height, width = lab_image.shape[:2]
    segmented = labels.reshape(height, width)
    image_center = np.array([width // 2, height // 2])
    cluster_data: list[dict[str, float | int | bool]] = []

    for cluster_id in range(4):
        cluster_mask = (segmented == cluster_id).astype(np.uint8) * 255
        analysis_mask = cv2.bitwise_and(cluster_mask, useful_area)
        area = cv2.countNonZero(analysis_mask)
        if area < 100:
            cluster_data.append({"id": cluster_id, "lightness": 0.0, "reject": True})
            continue

        lightness, channel_a, channel_b, _ = cv2.mean(lab_image, mask=analysis_mask)
        chroma = np.hypot(channel_a - 128, channel_b - 128)
        moments = cv2.moments(analysis_mask)
        centroid = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        )
        distance = np.linalg.norm(centroid - image_center)
        reject = distance > min(height, width) * 0.45 and lightness < 80 and chroma < 12
        if has_vignette and lightness < 35:
            reject = True
        cluster_data.append(
            {"id": cluster_id, "lightness": lightness, "reject": reject}
        )

    valid = sorted(
        (item for item in cluster_data if not item["reject"]),
        key=lambda item: float(item["lightness"]),
    )
    if not valid:
        valid = sorted(cluster_data, key=lambda item: float(item["lightness"]))

    core_id = int(valid[0]["id"])
    skin_lightness = float(valid[-1]["lightness"])
    core_lightness = float(valid[0]["lightness"])
    lesion_mask = (segmented == core_id).astype(np.uint8) * 255

    if len(valid) > 1:
        soft_id = int(valid[1]["id"])
        threshold = (core_lightness + skin_lightness) / 2
        if float(valid[1]["lightness"]) < threshold:
            lesion_mask = cv2.bitwise_or(
                lesion_mask, (segmented == soft_id).astype(np.uint8) * 255
            )

    rejected_ids = [int(item["id"]) for item in cluster_data if item["reject"]]
    return cv2.bitwise_and(lesion_mask, useful_area), rejected_ids


def _grabcut_refinement(
    lab_image: Image,
    initial_mask: Mask,
    useful_area: Mask,
    labels: NDArray[np.int64],
    rejected_ids: list[int],
) -> Mask:
    height, width = initial_mask.shape
    segmented = labels.reshape(height, width)
    grabcut_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[useful_area == 0] = cv2.GC_BGD
    for cluster_id in rejected_ids:
        grabcut_mask[segmented == cluster_id] = cv2.GC_BGD
    grabcut_mask[initial_mask == 255] = cv2.GC_PR_FGD

    try:
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        iterations = (
            1 if cv2.countNonZero(initial_mask) / (height * width) > 0.05 else 2
        )
        cv2.grabCut(
            lab_image,
            grabcut_mask,
            None,
            background,
            foreground,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
        result = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        result = initial_mask
    return cv2.bitwise_and(result, useful_area)


def _clean_mask(mask: Mask, useful_area: Mask) -> Mask:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    image_center = np.array([width // 2, height // 2])
    total_area = height * width
    cleaned = np.zeros_like(mask)

    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if not 200 <= area <= total_area * 0.7:
            continue
        component_mask = (labels == component).astype(np.uint8) * 255
        inside = cv2.countNonZero(cv2.bitwise_and(component_mask, useful_area))
        if inside / area < 0.8:
            continue
        distance = np.linalg.norm(centroids[component] - image_center)
        if distance < min(height, width) * 0.48 or area > 600:
            cleaned[labels == component] = 255

    if cv2.countNonZero(cleaned) == 0:
        return mask

    kernel_size = max(5, round(min(height, width) * 0.02))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return cv2.bitwise_and(filled, useful_area)


def segment_lesion(lab_image: Image, kmeans_pixels: NDArray[np.float32]) -> Mask:
    """Segment the probable lesion and return a binary mask."""
    height, width = lab_image.shape[:2]
    if len(kmeans_pixels) != height * width:
        raise ValueError("Pixel data does not match image dimensions")

    useful_area, has_vignette = _useful_image_area(lab_image)
    labels = _run_kmeans(kmeans_pixels, height, width)
    initial_mask, rejected = _select_lesion_clusters(
        lab_image, labels, useful_area, has_vignette
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    initial_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_CLOSE, kernel)
    refined = _grabcut_refinement(
        lab_image, initial_mask, useful_area, labels, rejected
    )
    return _clean_mask(refined, useful_area)
