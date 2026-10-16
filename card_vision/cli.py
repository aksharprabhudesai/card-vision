"""CLI entrypoint for card-vision."""

import argparse
import json
import sys
from pathlib import Path

import cv2

from card_vision.scanner import CardScanner


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        prog="card-vision",
        description="Extract and validate financial card attributes from images.",
    )
    parser.add_argument("command", choices=["scan"], help="Command to execute")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input card image")
    parser.add_argument(
        "--mask", action="store_true", default=True, help="Mask PAN digits in output"
    )
    parser.add_argument(
        "--no-mask", dest="mask", action="store_false", help="Expose raw PAN digits"
    )
    parser.add_argument("--yolo-model", type=str, default=None, help="Custom YOLO ONNX model path")
    parser.add_argument("--ocr-model", type=str, default=None, help="Custom OCR ONNX model path")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.stderr.write(f"Error: image not found at '{image_path}'\n")
        sys.exit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        sys.stderr.write(f"Error: could not read image '{image_path}' with OpenCV\n")
        sys.exit(1)

    scanner = CardScanner(
        yolo_model_path=args.yolo_model,
        ocr_model_path=args.ocr_model,
    )

    result = scanner.scan(image, mask_pan_flag=args.mask)

    if result is None:
        sys.stdout.write("No card detected in image.\n")
        sys.exit(0)

    if args.json:
        sys.stdout.write(json.dumps(result.model_dump(), indent=2) + "\n")
    else:
        sys.stdout.write("--- Scan Result ---\n")
        sys.stdout.write(f"Network:     {result.network.value}\n")
        sys.stdout.write(f"Card Class:  {result.card_class.value}\n")
        sys.stdout.write(f"PAN:         {result.pan or 'Not detected'}\n")
        sys.stdout.write(f"Expiry:      {result.expiry_date or 'Not detected'}\n")
        sys.stdout.write(f"Cardholder:  {result.cardholder_name or 'Not detected'}\n")
        sys.stdout.write(f"Luhn Valid:  {result.is_luhn_valid}\n")
        sys.stdout.write(f"Confidence:  {result.confidence:.2f}\n")


if __name__ == "__main__":
    main()
