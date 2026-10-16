"""End-to-end integration tests for the trained ONNX CardScanner pipeline."""

import random

import numpy as np

from card_vision import CardClass, CardScanner, NetworkScheme
from tools.generate_synthetic_cards import (
    CARD_H,
    CARD_W,
    apply_perspective_distortion,
    create_base_card_texture,
    draw_card_details,
    generate_luhn_pan,
)


def test_trained_onnx_end_to_end_pipeline() -> None:
    """Verify that bundled ONNX detector + OCR models accurately extract card attributes."""
    random.seed(123)
    np.random.seed(123)

    scanner = CardScanner()
    assert scanner.yolo_session is not None
    assert scanner.recognizer.session is not None

    test_cases = [
        (
            "Visa",
            "4111",
            16,
            "Debit",
            "12/28",
            "ALICE M SMITH",
            NetworkScheme.VISA,
            CardClass.DEBIT,
        ),
        (
            "Mastercard",
            "5105",
            16,
            "Credit",
            "08/30",
            "JOHN DOE",
            NetworkScheme.MASTERCARD,
            CardClass.CREDIT,
        ),
        (
            "Amex",
            "3782",
            15,
            "Credit",
            "05/29",
            "SARAH CONNOR",
            NetworkScheme.AMEX,
            CardClass.CREDIT,
        ),
        (
            "RuPay",
            "6521",
            16,
            "Credit",
            "11/27",
            "RAHUL SHARMA",
            NetworkScheme.RUPAY,
            CardClass.CREDIT,
        ),
        (
            "Discover",
            "6011",
            16,
            "Credit",
            "03/31",
            "DAVID BOWIE",
            NetworkScheme.DISCOVER,
            CardClass.CREDIT,
        ),
    ]

    for network_str, prefix, length, cls_str, expiry, holder, exp_net, exp_cls in test_cases:
        pan = generate_luhn_pan(prefix, length)
        card_img = create_base_card_texture(CARD_W, CARD_H)
        draw_card_details(card_img, pan, expiry, holder, network_str, cls_str)
        scene_bgr, _ = apply_perspective_distortion(card_img, 1280, 960)

        # 1. Test unmasked extraction
        res_unmasked = scanner.scan(scene_bgr, mask_pan_flag=False)
        assert res_unmasked is not None
        assert res_unmasked.pan == pan
        assert res_unmasked.is_luhn_valid is True
        assert res_unmasked.expiry_date == expiry
        assert res_unmasked.cardholder_name == holder
        assert res_unmasked.network == exp_net
        assert res_unmasked.card_class == exp_cls
        assert res_unmasked.confidence >= 0.85

        # 2. Test privacy-first masked extraction
        res_masked = scanner.scan(scene_bgr, mask_pan_flag=True)
        assert res_masked is not None
        assert res_masked.pan == res_masked.masked_pan
        assert res_masked.pan is not None
        assert res_masked.pan.endswith(pan[-4:])
        assert "*" in res_masked.pan


def test_cli_scan_command(tmp_path: object, monkeypatch: object, capsys: object) -> None:
    """Verify the card-vision CLI entrypoint executes and outputs JSON and formatted text."""
    from pathlib import Path

    import cv2
    import pytest

    from card_vision.cli import main

    assert isinstance(tmp_path, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    assert isinstance(capsys, pytest.CaptureFixture)

    random.seed(77)
    np.random.seed(77)
    pan = generate_luhn_pan("4111", 16)
    card_img = create_base_card_texture(CARD_W, CARD_H)
    draw_card_details(card_img, pan, "10/29", "JOHN DOE", "Visa", "Debit")
    scene_bgr, _ = apply_perspective_distortion(card_img, 1280, 960)

    img_path = tmp_path / "sample_card.jpg"
    cv2.imwrite(str(img_path), scene_bgr)

    # Test masked CLI output
    monkeypatch.setattr("sys.argv", ["card-vision", "scan", "--image", str(img_path), "--mask"])
    main()
    captured = capsys.readouterr()
    assert "Visa" in captured.out
    assert "Debit" in captured.out
    assert pan[-4:] in captured.out

    # Test JSON unmasked CLI output
    monkeypatch.setattr(
        "sys.argv", ["card-vision", "scan", "--image", str(img_path), "--no-mask", "--json"]
    )
    main()
    captured_json = capsys.readouterr()
    assert pan in captured_json.out
    assert '"is_luhn_valid": true' in captured_json.out
