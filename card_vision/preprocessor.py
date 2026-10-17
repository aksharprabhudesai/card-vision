"""Homography, deskewing, perspective rectification, and ROI cropping."""

import cv2
import numpy as np

from card_vision.models import BoundingBox

# ISO/IEC 7810 ID-1 standard dimensions: 85.60 mm x 53.98 mm (~1.5858 ratio)
# Standardized pixel dimension: 856 x 540 px
STD_CARD_WIDTH: int = 856
STD_CARD_HEIGHT: int = 540


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 (x, y) coordinates clockwise: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Args:
        pts: Numpy array of shape (4, 2).

    Returns:
        Ordered numpy array of shape (4, 2) dtype float32.
    """
    pts = pts.reshape((4, 2)).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top-left has the smallest sum; bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has the smallest difference (y - x or x - y)
    # Using np.diff(pts, axis=1): diff = y - x
    # TR: x is large, y is small -> y - x is minimum (most negative)
    # BL: x is small, y is large -> y - x is maximum (most positive)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_perspective_id1(
    image: np.ndarray,
    pts: np.ndarray,
    target_width: int = STD_CARD_WIDTH,
    target_height: int = STD_CARD_HEIGHT,
) -> np.ndarray:
    """Warp input card image to standard ISO/IEC 7810 ID-1 perspective (target_width x target_height).

    Args:
        image: Original input image (H, W, C) or (H, W).
        pts: 4 corner points array of shape (4, 2) in any order.
        target_width: Destination rectified width in pixels.
        target_height: Destination rectified height in pixels.

    Returns:
        Deskewed card image of shape (target_height, target_width, C).
    """
    rect = order_points(pts)

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    # Compute homography transformation matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return warped


def find_card_contour_corners(
    image: np.ndarray,
    bbox: BoundingBox | None = None,
) -> np.ndarray | None:
    """Fallback corner detector using edge detection and contour approximation.

    Used when YOLO detection provides bounding box only or for refinement.

    Args:
        image: Full image or cropped card image.
        bbox: Optional card bounding box to restrict edge search.

    Returns:
        Array of 4 corner points (shape: (4, 2)) or None if detection fails.
    """
    h_orig, w_orig = image.shape[:2]
    offset_x, offset_y = 0, 0

    if bbox is not None:
        x1, y1, x2, y2 = bbox.to_int_xyxy()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)
        crop = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        crop = image

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    candidate_masks = []
    # 1. Border background color difference segmentation (robust for cards on desk/uniform backgrounds)
    if crop.ndim == 3 and crop.shape[0] > 20 and crop.shape[1] > 20:
        border_pixels = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
        bg_med = np.median(border_pixels, axis=0).astype(np.float32)
        diff = np.linalg.norm(crop.astype(np.float32) - bg_med[None, None, :], axis=2)
        fg_mask = (diff > 18.0).astype(np.uint8) * 255
        fg_mask = cv2.morphologyEx(
            fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        candidate_masks.append(fg_mask)

    # 2. Multi-scale Canny edge maps
    for low, high in ((25, 80), (50, 150)):
        edges = cv2.Canny(blurred, low, high)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        candidate_masks.append(edges)

    for mask in candidate_masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:5]:
            if cv2.contourArea(cnt) < 5000:
                continue
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.03, 0.04, 0.05):
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape((4, 2)).astype(np.float32)
                    pts[:, 0] += offset_x
                    pts[:, 1] += offset_y

                    rect = order_points(pts)
                    w_top = float(np.linalg.norm(rect[1] - rect[0]))
                    w_bot = float(np.linalg.norm(rect[2] - rect[3]))
                    h_left = float(np.linalg.norm(rect[3] - rect[0]))
                    h_right = float(np.linalg.norm(rect[2] - rect[1]))

                    avg_w = (w_top + w_bot) / 2.0
                    avg_h = (h_left + h_right) / 2.0

                    if avg_h > 0:
                        aspect = avg_w / avg_h
                        if 1.1 <= aspect <= 2.2 and (avg_w * avg_h) > 5000:
                            return rect

    # Fallback to bbox corners if no 4-point contour found
    if bbox is not None:
        return np.array(
            [
                [bbox.xmin, bbox.ymin],
                [bbox.xmax, bbox.ymin],
                [bbox.xmax, bbox.ymax],
                [bbox.xmin, bbox.ymax],
            ],
            dtype=np.float32,
        )

    return None


def get_standard_rois(
    warped_card: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract standard ISO/IEC 7810 ID-1 Card ROIs from a rectified (856x540) card.

    Standard card layout relative positions:
    - PAN (Card Number): Horizontal strip centered vertically (~45% - 65% Y, 5% - 95% X)
    - Expiration Date: Typically bottom right/center (~62% - 78% Y, 40% - 75% X)
    - Cardholder Name: Typically lower third (~76% - 92% Y, 5% - 75% X)
    - Scheme Logo: Typically bottom right or top right (~70% - 95% Y, 72% - 96% X)

    Args:
        warped_card: Image of shape (540, 856, 3).

    Returns:
        Dictionary mapping ROI name to cropped image.
    """
    h, w = warped_card.shape[:2]

    # PAN Region: y from 42% to 65%, x from 5% to 95%
    pan_y1, pan_y2 = int(0.42 * h), int(0.65 * h)
    pan_x1, pan_x2 = int(0.05 * w), int(0.95 * w)

    # Expiry Region: y from 62% to 78%, x from 42% to 75%
    exp_y1, exp_y2 = int(0.62 * h), int(0.78 * h)
    exp_x1, exp_x2 = int(0.42 * w), int(0.75 * w)

    # Name Region: y from 75% to 92%, x from 5% to 75%
    name_y1, name_y2 = int(0.75 * h), int(0.92 * h)
    name_x1, name_x2 = int(0.05 * w), int(0.75 * w)

    # Logo Region: y from 70% to 95%, x from 70% to 96%
    logo_y1, logo_y2 = int(0.70 * h), int(0.95 * h)
    logo_x1, logo_x2 = int(0.70 * w), int(0.96 * w)

    return {
        "pan": warped_card[pan_y1:pan_y2, pan_x1:pan_x2],
        "expiry": warped_card[exp_y1:exp_y2, exp_x1:exp_x2],
        "name": warped_card[name_y1:name_y2, name_x1:name_x2],
        "logo": warped_card[logo_y1:logo_y2, logo_x1:logo_x2],
    }


def redact_pan_region(
    warped_card: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Mask/black-out the PAN region on a rectified card image for PCI-DSS compliance.

    Args:
        warped_card: Rectified card image.
        color: RGB/BGR tuple for the redacted rectangle.

    Returns:
        Redacted image copy.
    """
    redacted: np.ndarray = warped_card.copy()
    h, w = redacted.shape[:2]
    y1, y2 = int(0.42 * h), int(0.66 * h)
    x1, x2 = int(0.05 * w), int(0.95 * w)
    cv2.rectangle(redacted, (x1, y1), (x2, y2), color, -1)
    return redacted
