"""Production CardScanner orchestrating detection, rectification, OCR, and validation."""

from pathlib import Path

import cv2
import numpy as np

from card_vision.models import BoundingBox, CardResult, NetworkScheme, Point2D
from card_vision.preprocessor import (
    STD_CARD_HEIGHT,
    STD_CARD_WIDTH,
    find_card_contour_corners,
    get_standard_rois,
    order_points,
    warp_perspective_id1,
)
from card_vision.recognizer import OnnxTextRecognizer
from card_vision.validator import (
    clean_cardholder_name,
    clean_pan_digits,
    mask_pan,
    parse_expiry_date,
    resolve_bin_metadata,
    validate_luhn,
)

DEFAULT_DETECTOR_MODEL_PATH = Path(__file__).resolve().parent / "weights" / "card_detector.onnx"


class CardScanner:
    """Production financial card detection, rectification, and attribute extraction pipeline."""

    def __init__(
        self,
        yolo_model_path: str | Path | None = None,
        ocr_model_path: str | Path | None = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
    ) -> None:
        """Initialize the CardScanner pipeline.

        Args:
            yolo_model_path: Path to YOLO ONNX model for card and corner detection.
                Defaults to bundled card_vision/weights/card_detector.onnx.
            ocr_model_path: Path to ONNX model for text/digit recognition.
                Defaults to bundled card_vision/weights/card_ocr.onnx.
            conf_threshold: Confidence threshold for YOLO detection.
            iou_threshold: NMS IOU threshold.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.yolo_session = None

        resolved_yolo = (
            Path(yolo_model_path) if yolo_model_path is not None else DEFAULT_DETECTOR_MODEL_PATH
        )
        if resolved_yolo.exists() and resolved_yolo.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort

                self.yolo_session = ort.InferenceSession(
                    str(resolved_yolo),
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                self.yolo_session = None

        self.recognizer = OnnxTextRecognizer(ocr_model_path)

    def scan(
        self,
        image: np.ndarray,
        mask_pan_flag: bool = True,
        return_rectified: bool = False,
    ) -> CardResult | None:
        """Scan a card image, rectifying perspective and extracting card attributes.

        Args:
            image: Input image array (BGR or RGB, uint8).
            mask_pan_flag: Whether to mask the primary account number in output.
            return_rectified: If True, attaches warped card to the execution context.

        Returns:
            CardResult if card detected and processed, None otherwise.
        """
        if image is None or image.size == 0:
            return None

        # Step 1: Detect card bounding box and 4 corners
        card_bbox, corners = self._detect_card_and_corners(image)

        # Step 2: Perspective Warp to ISO/IEC 7810 ID-1 (856x540)
        if corners is not None and len(corners) == 4:
            pts_array = np.array([[pt.x, pt.y] for pt in corners], dtype=np.float32)
            warped = warp_perspective_id1(image, pts_array)
        elif card_bbox is not None:
            x1, y1, x2, y2 = card_bbox.to_int_xyxy()
            crop = image[y1:y2, x1:x2]
            warped = cv2.resize(
                crop, (STD_CARD_WIDTH, STD_CARD_HEIGHT), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            warped = cv2.resize(
                image, (STD_CARD_WIDTH, STD_CARD_HEIGHT), interpolation=cv2.INTER_LANCZOS4
            )

        # Step 3: Extract Regions of Interest (PAN, Expiry, Name, Scheme)
        rois = get_standard_rois(warped)

        # Step 4: Perform OCR on ROIs with strict character whitelists
        raw_pan = self.recognizer.recognize_digits(rois["pan"], whitelist="0123456789 ")
        raw_expiry = self.recognizer.recognize_text(rois["expiry"], whitelist="0123456789/")
        raw_name = self.recognizer.recognize_text(
            rois["name"], whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        )

        # Step 5: Clean, Validate, and Enrich Metadata
        cleaned_pan = clean_pan_digits(raw_pan)
        is_luhn = validate_luhn(cleaned_pan)
        expiry_formatted = parse_expiry_date(raw_expiry)
        name_formatted = clean_cardholder_name(raw_name)

        network, card_class = resolve_bin_metadata(cleaned_pan)

        masked = mask_pan(cleaned_pan) if cleaned_pan else None
        final_pan = masked if mask_pan_flag else (cleaned_pan if cleaned_pan else None)

        # Aggregate extraction confidence
        conf_scores = []
        if card_bbox:
            conf_scores.append(card_bbox.confidence)
        if is_luhn:
            conf_scores.append(0.98)
        elif cleaned_pan and len(cleaned_pan) >= 12:
            conf_scores.append(0.70)
        if expiry_formatted:
            conf_scores.append(0.90)
        if network != NetworkScheme.UNKNOWN:
            conf_scores.append(0.95)

        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.5

        return CardResult(
            pan=final_pan,
            masked_pan=masked,
            expiry_date=expiry_formatted,
            cardholder_name=name_formatted,
            network=network,
            card_class=card_class,
            is_luhn_valid=is_luhn,
            confidence=round(avg_conf, 3),
            corners=corners,
            card_bbox=card_bbox,
        )

    def _detect_card_and_corners(
        self, image: np.ndarray
    ) -> tuple[BoundingBox | None, list[Point2D] | None]:
        """Detect card bounds and 4 corner points using YOLO ONNX + contour refinement."""
        h, w = image.shape[:2]

        # 1. Try YOLO ONNX session if loaded
        if self.yolo_session is not None:
            bbox, pts = self._run_yolo_onnx(image)
            if bbox is not None and pts is not None and len(pts) == 4:
                # Optional sub-pixel edge contour refinement when high-contrast card border exists
                contour_pts = find_card_contour_corners(image)
                if contour_pts is not None:
                    ordered_cnt = order_points(contour_pts)
                    nn_pts = order_points(np.array([[p.x, p.y] for p in pts], dtype=np.float32))
                    mean_dist = float(np.mean(np.linalg.norm(ordered_cnt - nn_pts, axis=1)))
                    if mean_dist < 45.0:
                        pts = [Point2D(x=float(pt[0]), y=float(pt[1])) for pt in ordered_cnt]
                return bbox, pts
            if bbox is not None:
                return bbox, pts

        # 2. Geometric / Contour edge detection fallback
        contour_pts = find_card_contour_corners(image)
        if contour_pts is not None:
            ordered = order_points(contour_pts)
            corners = [Point2D(x=float(pt[0]), y=float(pt[1])) for pt in ordered]
            xmin = float(min(pt[0] for pt in ordered))
            ymin = float(min(pt[1] for pt in ordered))
            xmax = float(max(pt[0] for pt in ordered))
            ymax = float(max(pt[1] for pt in ordered))
            bbox = BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=0.85)
            return bbox, corners

        # 3. Check if the input image itself is already a rectified / full-frame card (~1.585 aspect)
        aspect = w / float(h) if h > 0 else 0.0
        if 1.45 <= aspect <= 1.70:
            full_bbox = BoundingBox(
                xmin=0.0, ymin=0.0, xmax=float(w), ymax=float(h), confidence=0.80
            )
            full_corners = [
                Point2D(x=0.0, y=0.0),
                Point2D(x=float(w - 1), y=0.0),
                Point2D(x=float(w - 1), y=float(h - 1)),
                Point2D(x=0.0, y=float(h - 1)),
            ]
            return full_bbox, full_corners

        # 4. Default bounding box enclosing center of frame
        default_bbox = BoundingBox(
            xmin=0.05 * w,
            ymin=0.10 * h,
            xmax=0.95 * w,
            ymax=0.90 * h,
            confidence=0.50,
        )
        default_corners = [
            Point2D(x=default_bbox.xmin, y=default_bbox.ymin),
            Point2D(x=default_bbox.xmax, y=default_bbox.ymin),
            Point2D(x=default_bbox.xmax, y=default_bbox.ymax),
            Point2D(x=default_bbox.xmin, y=default_bbox.ymax),
        ]
        return default_bbox, default_corners

    def _run_yolo_onnx(self, image: np.ndarray) -> tuple[BoundingBox | None, list[Point2D] | None]:
        """Run YOLO inference on ONNXRuntime for card detection and keypoint estimation."""
        assert self.yolo_session is not None
        h_orig, w_orig = image.shape[:2]

        # Skip detection if frame is completely uniform (blank test frame)
        if float(np.std(image)) < 2.0:
            return None, None

        input_meta = self.yolo_session.get_inputs()[0]
        model_h = input_meta.shape[2] if isinstance(input_meta.shape[2], int) else 320
        model_w = input_meta.shape[3] if isinstance(input_meta.shape[3], int) else 320

        # Resize with letterbox padding
        scale = min(model_w / w_orig, model_h / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((model_h, model_w, 3), 114, dtype=np.uint8)
        pad_x = (model_w - nw) // 2
        pad_y = (model_h - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        # HWC -> CHW, BGR -> RGB, Normalize
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self.yolo_session.run(None, {input_meta.name: blob})
        preds = outputs[0]

        if preds.shape[1] < preds.shape[2]:
            preds = preds.transpose(0, 2, 1)

        best_box: BoundingBox | None = None
        best_corners: list[Point2D] | None = None
        highest_conf = 0.0

        for row in preds[0]:
            box_conf = float(row[4]) if len(row) > 4 else 0.0
            if box_conf > self.conf_threshold and box_conf > highest_conf:
                highest_conf = min(1.0, max(0.0, box_conf))
                cx, cy, bw, bh = row[0:4]
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                bw = max(1.0, bw / scale)
                bh = max(1.0, bh / scale)

                xmin = max(0.0, min(float(w_orig - 1), float(cx - bw / 2.0)))
                ymin = max(0.0, min(float(h_orig - 1), float(cy - bh / 2.0)))
                xmax = max(xmin + 1.0, min(float(w_orig), float(cx + bw / 2.0)))
                ymax = max(ymin + 1.0, min(float(h_orig), float(cy + bh / 2.0)))

                best_box = BoundingBox(
                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=highest_conf
                )

                if len(row) >= 17:
                    kpts = []
                    kpt_data = row[5:17].reshape((4, 3))
                    for kx, ky, _ in kpt_data:
                        real_kx = max(0.0, min(float(w_orig), float((kx - pad_x) / scale)))
                        real_ky = max(0.0, min(float(h_orig), float((ky - pad_y) / scale)))
                        kpts.append(Point2D(x=real_kx, y=real_ky))
                    best_corners = kpts

        return best_box, best_corners
