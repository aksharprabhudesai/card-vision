"""Production REST API microservice for card-vision using FastAPI."""

import base64

import cv2
import numpy as np
from pydantic import BaseModel, Field

from card_vision import __version__
from card_vision.models import CardResult
from card_vision.scanner import CardScanner

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class Base64ScanRequest(BaseModel):
    """Payload schema for base64 encoded image scanning."""

    image_base64: str = Field(..., description="Base64 encoded image string (JPEG/PNG)")
    mask_pan: bool = Field(default=True, description="Whether to mask PAN in output")


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application instance."""
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required to run the server. "
            "Install it via 'pip install fastapi uvicorn python-multipart'"
        )

    app = FastAPI(
        title="card-vision API",
        version=__version__,
        description="Production-grade, privacy-first financial card scanning and attribute extraction engine.",
    )

    # Initialize shared singleton scanner
    scanner = CardScanner()

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, object]:
        """Health and readiness probe."""
        return {
            "status": "healthy",
            "version": __version__,
            "detector_loaded": scanner.yolo_session is not None,
            "recognizer_loaded": scanner.recognizer.session is not None,
        }

    @app.post(
        "/v1/scan",
        response_model=CardResult | None,
        tags=["Inference"],
        summary="Scan financial card from image upload",
    )
    async def scan_file(
        file: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)"),  # noqa: B008
        mask_pan: bool = Query(default=True, description="Mask sensitive PAN digits"),  # noqa: B008
    ) -> CardResult | None:
        """Process an uploaded card image file and return extracted attributes."""
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format.")

        result = scanner.scan(img, mask_pan_flag=mask_pan)
        return result

    @app.post(
        "/v1/scan-base64",
        response_model=CardResult | None,
        tags=["Inference"],
        summary="Scan financial card from Base64 string",
    )
    async def scan_base64(request: Base64ScanRequest) -> CardResult | None:
        """Process a Base64-encoded card image."""
        raw_b64 = request.image_base64
        if "," in raw_b64:
            # Strip data URI prefix (e.g. data:image/jpeg;base64,...)
            raw_b64 = raw_b64.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(raw_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to decode base64 image: {e}"
            ) from e

        if img is None:
            raise HTTPException(status_code=400, detail="Decoded bytes did not form a valid image.")

        result = scanner.scan(img, mask_pan_flag=request.mask_pan)
        return result

    return app
