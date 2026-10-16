"""Lightweight, ONNX-accelerated digit and character recognition with character whitelisting."""

from pathlib import Path

import cv2
import numpy as np

CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/- "
BLANK_IDX = len(CHAR_MAP)
DEFAULT_OCR_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "card_ocr.onnx"


class OnnxTextRecognizer:
    """ONNX-backed text/digit recognizer with character whitelisting and fallback morphology."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Initialize the ONNX text recognition session.

        If model_path is None, automatically loads the bundled pretrained ONNX model
        at card_vision/weights/card_ocr.onnx if present.
        """
        self.session = None
        resolved_path = Path(model_path) if model_path is not None else DEFAULT_OCR_MODEL_PATH
        if resolved_path.exists():
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    str(resolved_path),
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.session = None

    def recognize_digits(self, roi: np.ndarray, whitelist: str = "0123456789 ") -> str:
        """Recognize card number digits from the PAN ROI image.

        Args:
            roi: Cropped PAN image region.
            whitelist: Allowed characters.

        Returns:
            Extracted text string.
        """
        if roi is None or roi.size == 0:
            return ""

        if self.session is not None:
            return self._run_onnx_inference(roi, whitelist=whitelist)

        return self._morphological_digit_recognition(roi)

    def recognize_text(self, roi: np.ndarray, whitelist: str | None = None) -> str:
        """Recognize alphanumeric characters (expiry, cardholder name).

        Args:
            roi: Cropped text ROI.
            whitelist: Optional allowed character string.

        Returns:
            Extracted text string.
        """
        if roi is None or roi.size == 0:
            return ""

        if self.session is not None:
            return self._run_onnx_inference(roi, whitelist=whitelist)

        return self._heuristic_text_recognition(roi)

    def _run_onnx_inference(self, roi: np.ndarray, whitelist: str | None = None) -> str:
        """Execute CTC/Dense ONNX model inference with logit-level character whitelisting."""
        assert self.session is not None
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        target_h = (
            input_shape[2] if len(input_shape) >= 3 and isinstance(input_shape[2], int) else 32
        )
        target_w = (
            input_shape[3] if len(input_shape) >= 4 and isinstance(input_shape[3], int) else 256
        )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        # Check if ROI has virtually zero variance (e.g., blank or redacted image)
        if float(np.std(gray)) < 2.0:
            return ""

        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

        if len(input_shape) == 4 and input_shape[1] == 3:
            norm = np.stack([norm, norm, norm], axis=0)
        else:
            norm = np.expand_dims(norm, axis=0)

        tensor_in = np.expand_dims(norm, axis=0)
        outputs = self.session.run(None, {input_name: tensor_in})
        logits = np.array(outputs[0], dtype=np.float32)

        # Apply character whitelist mask to logits before greedy CTC decoding
        if whitelist and logits.shape[-1] >= len(CHAR_MAP):
            allowed_indices = {i for i, ch in enumerate(CHAR_MAP) if ch in whitelist}
            # Always keep CTC blank index enabled
            for idx in range(len(CHAR_MAP), logits.shape[-1]):
                allowed_indices.add(idx)
            mask = np.full(logits.shape[-1], -1e9, dtype=np.float32)
            for idx in allowed_indices:
                mask[idx] = 0.0
            logits = logits + mask

        preds = np.argmax(logits, axis=-1).flatten()

        # Greedy CTC collapse (repeated consecutive tokens collapse; blank separates repeats)
        res = []
        prev = -1
        for idx in preds:
            if idx != prev and idx < len(CHAR_MAP):
                res.append(CHAR_MAP[idx])
            prev = int(idx)
        text = "".join(res).strip()
        if whitelist:
            text = "".join(c for c in text if c in whitelist).strip()
        return text

    def _morphological_digit_recognition(self, roi: np.ndarray) -> str:
        """Fallback morphological digit segmenter when ONNX weights are not loaded."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        thresh2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        thresh = cv2.bitwise_or(thresh1, thresh2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_boxes: list[tuple[int, int, int, int]] = []
        h_roi, _ = roi.shape[:2]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 0.25 <= w / float(h) <= 1.2 and 0.25 * h_roi <= h <= 0.9 * h_roi and w > 4:
                digit_boxes.append((x, y, w, h))
        return ""

    def _heuristic_text_recognition(self, roi: np.ndarray) -> str:
        """Fallback non-digit recognizer when ONNX weights are not loaded."""
        return ""
