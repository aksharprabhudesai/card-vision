#!/usr/bin/env python3
"""Procedural OCR text line generator tailored for Financial Cards.

Mimics Farrington 7B, OCR-A, OCR-B, and laser-etched card typography to create
thousands of targeted text strips for training lightweight CRNN / ONNX text recognizers.
Generates:
- 16-digit PAN lines (with valid Luhn check digits and realistic font embossing)
- Expiration dates (MM/YY)
- Cardholder names (Latin uppercase)
- Paired label ground truth files (.txt / manifest.jsonl)
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAMPLE_SURNAMES = [
    "SMITH",
    "DOE",
    "KUMAR",
    "PATEL",
    "MUELLER",
    "ROSSI",
    "GARCIA",
    "TANAKA",
    "CHEN",
    "SILVA",
]
SAMPLE_FIRSTNAMES = [
    "ALICE",
    "JOHN",
    "VIKRAM",
    "PRIYA",
    "MARCUS",
    "ANNA",
    "EMMA",
    "LUCAS",
    "DAVID",
    "SARAH",
]


def get_ocr_font(size: int = 36) -> ImageFont.ImageFont:
    """Attempt to load standard monospaced fonts resembling Farrington 7B or OCR-A."""
    bundled_font = (
        Path(__file__).resolve().parent.parent / "card_vision" / "fonts" / "DejaVuSansMono.ttf"
    )
    candidates = [
        str(bundled_font),
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Courier.dfont",
        "/System/Library/Fonts/Menlo.ttc",
        "DejaVuSansMono.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_luhn_pan_digits(length: int = 16) -> str:
    """Generate valid 15 or 16 digit PAN string."""
    prefix = random.choice(["4", "51", "37", "6011", "6521"])
    remaining = length - len(prefix) - 1
    middle = "".join(str(random.randint(0, 9)) for _ in range(remaining))
    partial = prefix + middle

    digits = [int(d) for d in partial]
    checksum = 0
    parity = length % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    check_digit = (10 - (checksum % 10)) % 10
    return partial + str(check_digit)


def generate_expiry_text() -> str:
    """Generate realistic expiration date text."""
    prefix = random.choice(["", "VALID THRU ", "EXP ", "GOOD THRU ", "THRU "])
    mm = random.randint(1, 12)
    yy = random.randint(25, 33)
    sep = random.choice(["/", "-", " / "])
    return f"{prefix}{mm:02d}{sep}{yy:02d}"


def generate_name_text() -> str:
    """Generate realistic cardholder name string."""
    first = random.choice(SAMPLE_FIRSTNAMES)
    last = random.choice(SAMPLE_SURNAMES)
    if random.random() < 0.3:
        mid = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return f"{first} {mid} {last}"
    return f"{first} {last}"


def render_embossed_strip(
    text: str,
    strip_w: int = 320,
    strip_h: int = 48,
    font_size: int = 30,
    style: str = "embossed_silver",
) -> np.ndarray:
    """Render procedural text strip with realistic embossing, shadow, and surface noise."""
    # Base background (slate, gold, black, or blue card plastic)
    bg_color = random.choice(
        [
            (25, 30, 40),
            (35, 35, 40),
            (20, 45, 90),
            (70, 50, 20),
            (220, 220, 225),  # White plastic
        ]
    )
    canvas = np.full((strip_h, strip_w, 3), bg_color, dtype=np.uint8)

    # Add plastic grain texture
    noise = np.random.normal(0, 6, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = get_ocr_font(font_size)

    # Determine text colors based on background brightness
    is_dark_bg = sum(bg_color) < 400
    if is_dark_bg:
        text_color = random.choice([(240, 240, 245), (225, 190, 70), (200, 200, 200)])
        shadow_color = (15, 15, 20)
    else:
        text_color = random.choice([(20, 20, 25), (40, 40, 50), (10, 30, 80)])
        shadow_color = (255, 255, 255)

    # Position text vertically centered
    # Calculate bounding box using getbbox if available
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * (font_size * 0.6), font_size

    x = max(10, int((strip_w - tw) / 2) + random.randint(-5, 5))
    y = max(4, int((strip_h - th) / 2) + random.randint(-2, 2))

    if style == "embossed_silver":
        # Multi-layer emboss effect: shadow + highlight
        draw.text((x + 1, y + 2), text, font=font, fill=shadow_color)
        draw.text(
            (x - 1, y - 1), text, font=font, fill=(255, 255, 255) if is_dark_bg else (100, 100, 100)
        )
        draw.text((x, y), text, font=font, fill=text_color)
    else:
        # Flat print / laser-etched
        draw.text((x, y), text, font=font, fill=text_color)

    res = np.array(img)

    # Occasional blur or lighting degradation
    if random.random() < 0.25:
        res = cv2.GaussianBlur(res, (3, 3), 0)

    return res


def generate_ocr_dataset(output_dir: Path, total_samples: int = 100) -> None:
    """Generate synthetic OCR training strips with JSONL manifest."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = output_dir / "labels.jsonl"

    print(f"[*] Generating {total_samples} synthetic card OCR strips in: {output_dir}")

    records = []
    for i in range(total_samples):
        mode = random.choices(["pan", "expiry", "name"], weights=[0.5, 0.25, 0.25])[0]

        if mode == "pan":
            pan_raw = generate_luhn_pan_digits(length=random.choice([15, 16]))
            if len(pan_raw) == 15:
                text = f"{pan_raw[:4]} {pan_raw[4:10]} {pan_raw[10:]}"
            else:
                text = f"{pan_raw[:4]} {pan_raw[4:8]} {pan_raw[8:12]} {pan_raw[12:]}"
            gt_text = pan_raw
            w, h = 340, 50
        elif mode == "expiry":
            text = generate_expiry_text()
            gt_text = text
            w, h = 220, 44
        else:
            text = generate_name_text()
            gt_text = text
            w, h = 300, 44

        strip_bgr = render_embossed_strip(
            text=text,
            strip_w=w,
            strip_h=h,
            font_size=random.randint(24, 30),
            style=random.choice(["embossed_silver", "flat"]),
        )

        filename = f"ocr_strip_{i:06d}.png"
        file_path = img_dir / filename
        cv2.imwrite(str(file_path), strip_bgr)

        records.append(
            {
                "image": f"images/{filename}",
                "text": text,
                "ground_truth": gt_text,
                "category": mode,
                "width": w,
                "height": h,
            }
        )

    with open(manifest_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[+] Created {len(records)} OCR text samples and manifest at {manifest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic card OCR text strips.")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_ocr", help="Output path")
    parser.add_argument("--count", type=int, default=50, help="Number of text strips to generate")
    args = parser.parse_args()

    generate_ocr_dataset(Path(args.output_dir), total_samples=args.count)


if __name__ == "__main__":
    main()
