"""Measurement helpers for clot / edema / midline-shift masks.

Pure NumPy + OpenCV (no TensorFlow), so they can be unit-tested on their own.

Important conventions
---------------------
* Images are assumed to be axial slices shown in RADIOLOGICAL convention:
  the LEFT side of the image is the patient's RIGHT side. Pass
  ``radiological=False`` if your images use neurological convention.
* Physical sizes (mm, mm^2) are only reported when a real pixel spacing is
  given. JPG/PNG files do not store pixel spacing, so it must come from the
  original DICOM/NIfTI header. Without it, sizes are reported in pixels only.
* Midline shift is a DISTANCE (mm), not an area. Here it is estimated as the
  largest horizontal distance between the predicted midline-structure mask and
  the ideal midline (centre of the brain/skull outline). This is an
  approximation of the clinical measurement and must be validated against
  expert measurements before any claims are made.
"""

from __future__ import annotations

import numpy as np
import cv2


def brain_mask(gray: np.ndarray, background_threshold: int = 15) -> np.ndarray:
    """Rough head mask: low threshold against the dark background, close gaps,
    keep the largest component, fill holes.

    (Otsu is NOT used: on scans with a bright lesion it separates the lesion
    from the brain instead of the head from the background.)
    """
    gray = gray.astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Background level estimated from the image border (handles grey backgrounds).
    border = np.concatenate([blur[0], blur[-1], blur[:, 0], blur[:, -1]])
    level = max(background_threshold, float(np.median(border)) + background_threshold)
    th = (blur > level).astype(np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def ideal_midline_x(head: np.ndarray) -> float | None:
    """Ideal midline = horizontal centre of the head outline's bounding box."""
    xs = np.where(head.any(axis=0))[0]
    if xs.size == 0:
        return None
    return (xs.min() + xs.max()) / 2.0


def area(mask: np.ndarray, pixel_spacing_mm: float | None = None) -> dict:
    """Area of a binary mask in pixels, and in mm^2 if pixel spacing is known."""
    px = int(np.count_nonzero(mask))
    mm2 = px * pixel_spacing_mm ** 2 if pixel_spacing_mm else None
    return {"pixels": px, "mm2": mm2}


def side(mask: np.ndarray, midline_x: float | None, radiological: bool = True) -> str:
    """Which side of the PATIENT the region is mostly on."""
    xs = np.where(mask)[1]
    if xs.size == 0 or midline_x is None:
        return "None detected"
    image_left = np.count_nonzero(xs < midline_x)
    image_right = xs.size - image_left
    if image_left == image_right:
        return "Midline / cannot determine"
    on_image_left = image_left > image_right
    # Radiological convention: image left = patient right.
    patient_right = on_image_left if radiological else not on_image_left
    return "Patient's right hemisphere" if patient_right else "Patient's left hemisphere"


def midline_shift(midline_mask: np.ndarray, ideal_x: float | None,
                  pixel_spacing_mm: float | None = None, min_pixels: int = 20) -> dict:
    """Largest horizontal distance from the ideal midline to the predicted midline structure."""
    ys, xs = np.where(midline_mask)
    if xs.size < min_pixels or ideal_x is None:
        return {"pixels": None, "mm": None, "direction": "not measurable"}
    deviation = xs - ideal_x
    i = int(np.argmax(np.abs(deviation)))
    px = float(abs(deviation[i]))
    direction = "towards image left" if deviation[i] < 0 else "towards image right"
    mm = px * pixel_spacing_mm if pixel_spacing_mm else None
    return {"pixels": px, "mm": mm, "direction": direction}


def lesion_fraction(clot: np.ndarray, edema: np.ndarray, head: np.ndarray) -> float | None:
    """Percent of the head area covered by clot OR edema (overlap counted once)."""
    head_px = np.count_nonzero(head)
    if head_px == 0:
        return None
    lesion = np.logical_or(clot, edema) & head.astype(bool)
    return 100.0 * np.count_nonzero(lesion) / head_px
