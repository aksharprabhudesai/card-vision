"""Batch image directory processing with structured JSON output."""

import argparse
import json
from pathlib import Path

import cv2

from card_vision import CardScanner


def process_directory(input_dir: Path, output_json: Path, mask_pan: bool = True) -> None:
    scanner = CardScanner()
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]

    print(f"[*] Found {len(image_files)} image files in {input_dir}")
    records = []
    success_count = 0

    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = scanner.scan(img, mask_pan_flag=mask_pan)
        if result:
            success_count += 1
            data = result.model_dump()
            data["filename"] = img_path.name
            records.append(data)
            print(
                f"[{i}/{len(image_files)}] OK: {img_path.name} -> {result.network.value} ({result.masked_pan})"
            )
        else:
            print(f"[{i}/{len(image_files)}] NO CARD: {img_path.name}")

    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)

    print(
        f"[+] Processed {len(image_files)} cards. Detected: {success_count}. Saved to {output_json}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process cards in directory.")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory")
    parser.add_argument(
        "--output", "-o", type=str, default="batch_results.json", help="Output JSON path"
    )
    parser.add_argument("--no-mask", dest="mask", action="store_false", help="Expose unmasked PAN")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output), mask_pan=args.mask)


if __name__ == "__main__":
    main()
