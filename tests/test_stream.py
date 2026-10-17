"""Unit tests for StreamCardScanner temporal aggregation and voting."""

from typing import Any

import numpy as np

from card_vision.models import CardClass, CardResult, NetworkScheme, Point2D
from card_vision.stream import StreamCardScanner


def test_stream_temporal_consensus_voting(monkeypatch: Any) -> None:
    scanner = StreamCardScanner(history_size=5, min_consistent_frames=3)

    res1 = CardResult(
        pan="4111222233331111",
        masked_pan="**** **** **** 1111",
        is_luhn_valid=True,
        network=NetworkScheme.VISA,
        card_class=CardClass.CREDIT,
        confidence=0.80,
    )

    def mock_scan(
        image: np.ndarray, mask_pan_flag: bool = True, return_rectified: bool = False
    ) -> CardResult | None:
        return res1

    monkeypatch.setattr(scanner.scanner, "scan", mock_scan)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    r1, stable1 = scanner.process_frame(dummy_frame)
    assert stable1 is False

    r2, stable2 = scanner.process_frame(dummy_frame)
    assert stable2 is False

    # Frame 3: Consistent 3rd frame reaching consensus
    r3, stable3 = scanner.process_frame(dummy_frame)
    assert stable3 is True
    assert r3 is not None
    assert r3.pan == "4111222233331111"
    assert r3.confidence > 0.80


def test_stream_corner_smoothing(monkeypatch: Any) -> None:
    scanner = StreamCardScanner(smoothing_factor=0.5)

    corners_f1 = [
        Point2D(x=10.0, y=10.0),
        Point2D(x=90.0, y=10.0),
        Point2D(x=90.0, y=90.0),
        Point2D(x=10.0, y=90.0),
    ]
    corners_f2 = [
        Point2D(x=20.0, y=20.0),
        Point2D(x=100.0, y=20.0),
        Point2D(x=100.0, y=100.0),
        Point2D(x=20.0, y=100.0),
    ]

    res1 = CardResult(pan="4111222233331111", is_luhn_valid=True, corners=corners_f1)
    res2 = CardResult(pan="4111222233331111", is_luhn_valid=True, corners=corners_f2)

    current_res = res1

    def mock_scan(image: np.ndarray, **kwargs: Any) -> CardResult | None:
        return current_res

    monkeypatch.setattr(scanner.scanner, "scan", mock_scan)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    scanner.process_frame(dummy_frame)

    current_res = res2
    r2, _ = scanner.process_frame(dummy_frame)
    assert r2 is not None and r2.corners is not None
    # Expected EMA for top-left: 0.5 * 20 + 0.5 * 10 = 15.0
    assert abs(r2.corners[0].x - 15.0) < 1e-4
    assert abs(r2.corners[0].y - 15.0) < 1e-4
