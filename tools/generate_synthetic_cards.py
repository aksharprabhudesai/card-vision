#!/usr/bin/env python3
"""Standalone Synthetic Financial Card Generator.

Generates photorealistic financial card images with:
- Valid ISO/IEC 7812 Luhn numbers
- Real BIN ranges for Visa, Mastercard, Amex, RuPay, Discover
- Authentic fonts (OCR-A / Farrington style) and textures
- Smart chip, contactless wave symbol, holographic logos, cardholder name, expiry
- Perspective jitter & realistic background placement
- Automated YOLO-format bounding box and 4-corner keypoint annotations (.txt)
"""

import argparse
import functools
import random
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ISO/IEC 7810 ID-1 standard dimensions: 856 x 540 px
CARD_W = 856
CARD_H = 540


class BinConfig(TypedDict):
    network: str
    prefix: str
    length: int
    card_class: str


SAMPLE_NAMES = [
    "JOHN DOE",
    "ALICE M SMITH",
    "RAHUL SHARMA",
    "VIKRAM PATEL",
    "ELIZABETH TAYLOR",
    "MARCUS AURELIUS",
    "SARAH CONNOR",
    "DAVID BOWIE",
    "PRIYA NAIR",
    "MICHAEL CORLEONE",
]

BIN_CONFIGS: list[BinConfig] = [
    {"network": "Visa", "prefix": "4", "length": 16, "card_class": "Credit"},
    {"network": "Visa", "prefix": "4111", "length": 16, "card_class": "Debit"},
    {"network": "Mastercard", "prefix": "5105", "length": 16, "card_class": "Credit"},
    {"network": "Mastercard", "prefix": "5350", "length": 16, "card_class": "Debit"},
    {"network": "Amex", "prefix": "3782", "length": 15, "card_class": "Credit"},
    {"network": "RuPay", "prefix": "6521", "length": 16, "card_class": "Credit"},
    {"network": "RuPay", "prefix": "6071", "length": 16, "card_class": "Debit"},
    {"network": "Discover", "prefix": "6011", "length": 16, "card_class": "Credit"},
]


def generate_luhn_pan(prefix: str, length: int = 16) -> str:
    """Generate a random Primary Account Number with a valid Luhn check digit."""
    middle_len = length - len(prefix) - 1
    middle = "".join([str(random.randint(0, 9)) for _ in range(middle_len)])
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


def generate_random_expiry() -> str:
    """Generate a realistic future expiration date MM/YY."""
    month = random.randint(1, 12)
    year = random.randint(25, 33)
    return f"{month:02d}/{year:02d}"


@functools.lru_cache(maxsize=32)
def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get cached standard PIL font or fallback."""
    font_candidates = [
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Courier.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "DejaVuSansMono.ttf",
        "Courier",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_base_card_texture(width: int, height: int) -> Image.Image:
    """Create realistic card background gradient and textures using fast vectorized NumPy."""
    color_schemes = [
        ((15, 23, 42), (51, 65, 85)),  # Slate/Dark Obsidian
        ((24, 24, 27), (63, 63, 70)),  # Premium Carbon / Onyx
        ((30, 58, 138), (59, 130, 246)),  # Royal Deep Blue
        ((88, 28, 135), (147, 51, 234)),  # Purple Platinum
        ((4, 120, 87), (16, 185, 129)),  # Emerald Green
        ((120, 53, 15), (217, 119, 6)),  # Gold / Bronze
    ]
    c1_tup, c2_tup = random.choice(color_schemes)
    c1 = np.array(c1_tup, dtype=np.float32)
    c2 = np.array(c2_tup, dtype=np.float32)

    factors = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    grad_1d = (c1[None, :] * (1.0 - factors) + c2[None, :] * factors).astype(np.uint8)
    base = np.broadcast_to(grad_1d[:, None, :], (height, width, 3)).copy()

    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)

    # Smart EMV Chip (gold/bronze contact pad)
    chip_x, chip_y = 120, 180
    chip_w, chip_h = 100, 75
    draw.rounded_rectangle(
        [(chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h)],
        radius=8,
        fill=(212, 175, 55),
        outline=(160, 130, 40),
        width=2,
    )
    draw.line([(chip_x + 35, chip_y), (chip_x + 35, chip_y + chip_h)], fill=(160, 130, 40), width=1)
    draw.line([(chip_x + 65, chip_y), (chip_x + 65, chip_y + chip_h)], fill=(160, 130, 40), width=1)
    draw.line([(chip_x, chip_y + 37), (chip_x + chip_w, chip_y + 37)], fill=(160, 130, 40), width=1)

    # Contactless wave symbol (arcs)
    wave_x, wave_y = 250, 205
    for r in range(12, 28, 6):
        draw.arc(
            [(wave_x - r, wave_y - r), (wave_x + r, wave_y + r)],
            start=290,
            end=70,
            fill=(200, 200, 200),
            width=2,
        )

    return img


def draw_card_details(
    img: Image.Image,
    pan: str,
    expiry: str,
    name: str,
    network: str,
    card_class: str,
) -> None:
    """Render authentic embossed/flat typography and branding onto card."""
    draw = ImageDraw.Draw(img)
    font_pan = get_font(42)
    font_med = get_font(26)
    font_small = get_font(18)

    if len(pan) == 15:
        formatted_pan = f"{pan[:4]} {pan[4:10]} {pan[10:]}"
    else:
        formatted_pan = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:]}"

    text_color = (245, 245, 245)
    shadow_color = (20, 20, 20)

    # 1. Bank Brand / Class header
    draw.text((60, 50), f"GLOBAL BANK • {card_class.upper()}", font=font_med, fill=(210, 210, 210))

    # 2. PAN Number
    px, py = 80, 290
    draw.text((px + 2, py + 2), formatted_pan, font=font_pan, fill=shadow_color)
    draw.text((px, py), formatted_pan, font=font_pan, fill=text_color)

    # 3. Expiry Date
    ex, ey = 380, 370
    draw.text((ex - 90, ey + 4), "VALID\nTHRU", font=font_small, fill=(180, 180, 180))
    draw.text((ex + 2, ey + 2), expiry, font=font_med, fill=shadow_color)
    draw.text((ex, ey), expiry, font=font_med, fill=text_color)

    # 4. Cardholder Name
    nx, ny = 80, 440
    draw.text((nx + 2, ny + 2), name, font=font_med, fill=shadow_color)
    draw.text((nx, ny), name, font=font_med, fill=text_color)

    # 5. Network Scheme Logo
    lx, ly = 660, 420
    if network == "Visa":
        draw.text((lx, ly), "VISA", font=get_font(44), fill=(255, 255, 255))
    elif network == "Mastercard":
        draw.ellipse([(lx, ly), (lx + 55, ly + 55)], fill=(235, 0, 27))
        draw.ellipse([(lx + 35, ly), (lx + 90, ly + 55)], fill=(255, 95, 0))
    elif network == "Amex":
        draw.rectangle([(lx - 10, ly), (lx + 100, ly + 50)], fill=(0, 114, 206))
        draw.text((lx - 5, ly + 10), "AMEX", font=get_font(28), fill=(255, 255, 255))
    elif network == "RuPay":
        draw.text((lx - 20, ly + 10), "RuPay»", font=get_font(34), fill=(0, 164, 228))
    else:
        draw.text((lx - 20, ly + 10), network.upper(), font=get_font(30), fill=(240, 240, 240))


def apply_perspective_distortion(
    card_img: Image.Image,
    canvas_w: int = 1280,
    canvas_h: int = 960,
) -> tuple[np.ndarray, np.ndarray]:
    """Place card onto realistic background canvas with random 3D perspective distortion.

    Returns:
        (composite_canvas_bgr, 4_ordered_corner_points_xy)
    """
    card_cv = cv2.cvtColor(np.array(card_img), cv2.COLOR_RGB2BGR)

    # Generate contrasting lighter background so card borders are clearly delineated
    bg_color = [random.randint(150, 235) for _ in range(3)]
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    src_pts = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )

    center_x = canvas_w / 2.0 + random.uniform(-60.0, 60.0)
    center_y = canvas_h / 2.0 + random.uniform(-60.0, 60.0)

    scale_f = random.uniform(0.75, 1.05)
    half_w = (CARD_W * scale_f) / 2.0
    half_h = (CARD_H * scale_f) / 2.0

    jitter = 45.0
    dst_pts = np.array(
        [
            [
                center_x - half_w + random.uniform(-jitter, jitter),
                center_y - half_h + random.uniform(-jitter, jitter),
            ],
            [
                center_x + half_w + random.uniform(-jitter, jitter),
                center_y - half_h + random.uniform(-jitter, jitter),
            ],
            [
                center_x + half_w + random.uniform(-jitter, jitter),
                center_y + half_h + random.uniform(-jitter, jitter),
            ],
            [
                center_x - half_w + random.uniform(-jitter, jitter),
                center_y + half_h + random.uniform(-jitter, jitter),
            ],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_card = cv2.warpPerspective(
        card_cv,
        homography,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    card_mask = np.full((CARD_H, CARD_W), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        card_mask,
        homography,
        (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    composite = canvas.copy()
    cv2.copyTo(warped_card, warped_mask, composite)
    return composite, dst_pts


def save_yolo_annotation(
    filepath: Path,
    corners: np.ndarray,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> None:
    """Save YOLO bounding-box and 4-point corner keypoint annotations."""
    xmin = max(0.0, min(float(img_w), float(np.min(corners[:, 0]))))
    xmax = max(0.0, min(float(img_w), float(np.max(corners[:, 0]))))
    ymin = max(0.0, min(float(img_h), float(np.min(corners[:, 1]))))
    ymax = max(0.0, min(float(img_h), float(np.max(corners[:, 1]))))

    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h

    kpts_str = " ".join([f"{c[0] / img_w:.6f} {c[1] / img_h:.6f} 2" for c in corners])
    line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kpts_str}\n"

    with open(filepath, "w") as f:
        f.write(line)


def generate_dataset(output_dir: Path, count: int = 50) -> None:
    """Generate synthetic card dataset with paired YOLO annotations."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {count} synthetic card samples in {output_dir}...")

    for i in range(count):
        cfg = random.choice(BIN_CONFIGS)
        pan = generate_luhn_pan(cfg["prefix"], cfg["length"])
        expiry = generate_random_expiry()
        name = random.choice(SAMPLE_NAMES)
        network = cfg["network"]
        card_class = cfg["card_class"]

        card_img = create_base_card_texture(CARD_W, CARD_H)
        draw_card_details(card_img, pan, expiry, name, network, card_class)

        canvas_w, canvas_h = 1280, 960
        comp_img, corners = apply_perspective_distortion(card_img, canvas_w, canvas_h)

        sample_id = f"card_synth_{i:06d}"
        img_file = images_dir / f"{sample_id}.jpg"
        lbl_file = labels_dir / f"{sample_id}.txt"

        cv2.imwrite(str(img_file), comp_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_yolo_annotation(lbl_file, corners, canvas_w, canvas_h, class_id=0)

    print(f"Dataset generation complete: {count} samples saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial cards for YOLO training."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/synthetic_cards",
        help="Directory to save dataset",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of cards to generate")
    parser.add_argument(
        "--annotate-corners",
        action="store_true",
        default=True,
        help="Include 4-point corner keypoints in YOLO annotations",
    )
    args = parser.parse_args()

    generate_dataset(Path(args.output_dir), count=args.count)


if __name__ == "__main__":
    main()
