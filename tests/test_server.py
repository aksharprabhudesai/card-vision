"""Unit tests for FastAPI REST API endpoints."""

import base64

import cv2
import numpy as np
import pytest

from card_vision.server import FASTAPI_AVAILABLE, create_app


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_server_health_and_scan_endpoints() -> None:
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # 1. Test /health
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data

    # 2. Test /v1/scan-base64 with dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode(".jpg", img)
    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")

    scan_res = client.post("/v1/scan-base64", json={"image_base64": b64_str, "mask_pan": True})
    assert scan_res.status_code == 200
