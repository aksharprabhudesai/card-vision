"""Minimal Quickstart Example for card-vision."""

import sys
from pathlib import Path

import cv2

from card_vision import CardScanner


def main() -> None:
    # 1. Initialize scanner (bundled ONNX models are auto-loaded)
    scanner = CardScanner()

    # 2. Provide an image path (or use default test sample)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    p = Path(image_path)
    if not p.exists():
        print(f"File not found: {p}. Please pass a valid image path.")
        return

    img = cv2.imread(str(p))
    if img is None:
        print(f"Could not load image: {p}")
        return

    # 3. Scan card with privacy-first masking enabled by default
    result = scanner.scan(img, mask_pan_flag=True)

    if result is None:
        print("No financial card detected in image.")
        return

    # 4. Access verified structured fields
    print("=== Card Extraction Summary ===")
    print(f"Payment Network:  {result.network.value}")
    print(f"Account Class:    {result.card_class.value}")
    print(f"Masked PAN:       {result.masked_pan}")
    print(f"Expiration Date:  {result.expiry_date}")
    print(f"Cardholder Name:  {result.cardholder_name}")
    print(f"Luhn Validated:   {result.is_luhn_valid}")
    print(f"Model Confidence: {result.confidence:.2%}")


if __name__ == "__main__":
    main()
