from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
PROCESSED_IMAGE_DIR = BASE_DIR / "processed_images"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_KERNEL_SIZE = (5, 5)
CANNY_SIGMA = 0.33
FALLBACK_CANNY_THRESHOLD_LOW = 40
FALLBACK_CANNY_THRESHOLD_HIGH = 120
MORPH_KERNEL_SIZE = 3
MIN_CONTOUR_AREA_RATIO = 0.0005
FALLBACK_MIN_CONTOUR_AREA_RATIO = 0.00005
MIN_CONTOUR_PERIMETER = 30.0
MAX_FALLBACK_CONTOURS = 80
CONTOUR_LINE_WIDTH = 2


def _compute_canny_thresholds(image: np.ndarray) -> tuple[int, int]:
    median_intensity = float(np.median(image))
    lower = int(max(0, (1.0 - CANNY_SIGMA) * median_intensity))
    upper = int(min(255, (1.0 + CANNY_SIGMA) * median_intensity))

    if lower == upper:
        lower = max(0, lower - 20)
        upper = min(255, upper + 20)

    return lower, upper


def _build_edge_overlay(source_image: np.ndarray) -> tuple[np.ndarray, int, str]:
    grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    enhanced = clahe.apply(grayscale)
    blurred = cv2.GaussianBlur(enhanced, GAUSSIAN_KERNEL_SIZE, 0)

    threshold_low, threshold_high = _compute_canny_thresholds(blurred)
    auto_edges = cv2.Canny(blurred, threshold_low, threshold_high)
    fallback_edges = cv2.Canny(
        blurred,
        FALLBACK_CANNY_THRESHOLD_LOW,
        FALLBACK_CANNY_THRESHOLD_HIGH,
    )
    edges = cv2.bitwise_or(auto_edges, fallback_edges)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned_edges = cv2.dilate(cleaned_edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        cleaned_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = source_image.shape[0] * source_image.shape[1]
    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    filtered_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= min_area
    ]

    if not filtered_contours:
        fallback_min_area = image_area * FALLBACK_MIN_CONTOUR_AREA_RATIO
        fallback_contours = [
            contour for contour in contours
            if (
                cv2.contourArea(contour) >= fallback_min_area
                and cv2.arcLength(contour, True) >= MIN_CONTOUR_PERIMETER
            )
        ]
        fallback_contours.sort(key=cv2.contourArea, reverse=True)
        filtered_contours = fallback_contours[:MAX_FALLBACK_CONTOURS]

    overlay = source_image.copy()

    if filtered_contours:
        cv2.drawContours(
            overlay,
            filtered_contours,
            contourIdx=-1,
            color=(0, 255, 0),
            thickness=CONTOUR_LINE_WIDTH,
        )
        return overlay, len(filtered_contours), "contours"

    overlay[cleaned_edges > 0] = (0, 255, 0)
    edge_pixel_count = int((cleaned_edges > 0).sum())
    return overlay, edge_pixel_count, "edge_pixels"


def analyze_image(image_path: str, sensor_payload: dict | None = None) -> dict:
    """
    Run a simple edge-detection pass and save a green-edge visualization.
    """
    if not image_path:
        return {
            "summary": "No image received from module4 yet.",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_image_path": "",
        }

    source_path = Path(image_path)

    if not source_path.exists():
        return {
            "summary": f"Image file not found: {source_path}",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_image_path": "",
        }

    source_image = cv2.imread(str(source_path))

    if source_image is None:
        return {
            "summary": f"Failed to load image: {source_path.name}",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_image_path": "",
        }

    overlay, highlight_count, highlight_mode = _build_edge_overlay(source_image)

    PROCESSED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_IMAGE_DIR / f"{source_path.stem}_edges.jpg"

    if not cv2.imwrite(str(processed_path), overlay):
        return {
            "summary": f"Failed to save processed image for {source_path.name}",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "processed_image_path": "",
        }

    mic_rms = 0.0 if not sensor_payload else float(sensor_payload.get("mic_rms", 0.0))
    acc_rms = 0.0 if not sensor_payload else float(sensor_payload.get("acc_rms", 0.0))
    gyro_rms = 0.0 if not sensor_payload else float(sensor_payload.get("gyro_rms", 0.0))

    return {
        "summary": (
            f"Edge detection completed for {source_path.name}. "
            f"Highlighted {highlight_mode}={highlight_count}. "
            f"sensor context: mic_rms={mic_rms:.2f}, acc_rms={acc_rms:.4f}, gyro_rms={gyro_rms:.2f}"
        ),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "processed_image_path": str(processed_path),
    }
