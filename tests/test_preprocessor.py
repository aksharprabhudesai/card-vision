"""Unit tests for geometric transformations, homography, and card scanning."""

import numpy as np

from card_vision.preprocessor import (
    STD_CARD_HEIGHT,
    STD_CARD_WIDTH,
    get_standard_rois,
    order_points,
    redact_pan_region,
    warp_perspective_id1,
)
from card_vision.scanner import CardScanner


def test_order_points() -> None:
    # Unordered points: [BR, TL, BL, TR]
    pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
    ordered = order_points(pts)

    # Expected: TL (0,0), TR (100, 0), BR (100, 100), BL (0, 100)
    np.testing.assert_allclose(ordered[0], [0, 0])
    np.testing.assert_allclose(ordered[1], [100, 0])
    np.testing.assert_allclose(ordered[2], [100, 100])
    np.testing.assert_allclose(ordered[3], [0, 100])


def test_warp_perspective_id1() -> None:
    # Create synthetic test canvas
    canvas = np.zeros((600, 900, 3), dtype=np.uint8)
    canvas[100:500, 100:800] = 255

    pts = np.array([[100, 100], [800, 100], [800, 500], [100, 500]], dtype=np.float32)
    warped = warp_perspective_id1(
        canvas, pts, target_width=STD_CARD_WIDTH, target_height=STD_CARD_HEIGHT
    )

    assert warped.shape == (STD_CARD_HEIGHT, STD_CARD_WIDTH, 3)
    assert warped.mean() > 200  # Rectified white area


def test_get_standard_rois() -> None:
    card = np.zeros((STD_CARD_HEIGHT, STD_CARD_WIDTH, 3), dtype=np.uint8)
    rois = get_standard_rois(card)

    assert "pan" in rois
    assert "expiry" in rois
    assert "name" in rois
    assert "logo" in rois
    assert rois["pan"].shape[0] > 0
    assert rois["pan"].shape[1] > 0


def test_redact_pan_region() -> None:
    card = np.full((STD_CARD_HEIGHT, STD_CARD_WIDTH, 3), 255, dtype=np.uint8)
    redacted = redact_pan_region(card, color=(0, 0, 0))

    # Center where PAN resides should now be black (0, 0, 0)
    center_y = int(STD_CARD_HEIGHT * 0.5)
    center_x = int(STD_CARD_WIDTH * 0.5)
    np.testing.assert_array_equal(redacted[center_y, center_x], [0, 0, 0])


def test_card_scanner_mock_execution() -> None:
    scanner = CardScanner()
    # Provide a blank frame
    test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    result = scanner.scan(test_frame)

    assert result is not None
    assert result.card_bbox is not None
    assert result.network is not None
