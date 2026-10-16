#!/usr/bin/env python3
"""Train and export production ONNX models for card-vision.

Trains two lightweight, high-precision neural networks and exports them to ONNX:
1. card_detector.onnx: YOLO-Pose compatible card bounding-box & 4-corner keypoint detector
   Input: (1, 3, 320, 320) RGB float32 [0, 1]
   Output: (1, 17, 100) [cx, cy, w, h, conf, x1, y1, v1, x2, y2, v2, x3, y3, v3, x4, y4, v4]
2. card_ocr.onnx: CRNN / Conv-CTC sequence recognizer for PAN, Expiry, and Cardholder Name
   Input: (1, 1, 32, 256) Grayscale float32 [-1, 1]
   Output: (1, 64, 40) CTC logits over CHAR_MAP + blank
"""

import argparse
import random
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from card_vision.preprocessor import get_standard_rois, order_points, warp_perspective_id1
from card_vision.recognizer import BLANK_IDX, CHAR_MAP
from card_vision.scanner import CardScanner
from tools.generate_ocr_strips import SAMPLE_FIRSTNAMES, SAMPLE_SURNAMES
from tools.generate_synthetic_cards import (
    BIN_CONFIGS,
    CARD_H,
    CARD_W,
    SAMPLE_NAMES,
    apply_perspective_distortion,
    create_base_card_texture,
    draw_card_details,
    generate_luhn_pan,
    generate_random_expiry,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CardDetectorNet(nn.Module):
    """Lightweight YOLO-Pose compatible 10x10 spatial grid card & 4-corner detector."""

    def __init__(self, img_size: int = 320, grid_size: int = 10) -> None:
        super().__init__()
        self.img_size = float(img_size)
        self.grid_size = grid_size

        # Feature backbone: 320 -> 160 -> 80 -> 40 -> 20 -> 10
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_fc = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(160, 13, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        glob = self.global_fc(self.global_pool(feat))
        feat = feat + glob
        raw = self.head(feat)  # (B, 13, 10, 10)
        b = raw.shape[0]
        raw = raw.flatten(2)  # (B, 13, 100)

        cx = torch.sigmoid(raw[:, 0:1, :]) * self.img_size
        cy = torch.sigmoid(raw[:, 1:2, :]) * self.img_size
        bw = torch.sigmoid(raw[:, 2:3, :]) * self.img_size
        bh = torch.sigmoid(raw[:, 3:4, :]) * self.img_size
        conf = torch.sigmoid(raw[:, 4:5, :])

        x1 = torch.sigmoid(raw[:, 5:6, :]) * self.img_size
        y1 = torch.sigmoid(raw[:, 6:7, :]) * self.img_size
        x2 = torch.sigmoid(raw[:, 7:8, :]) * self.img_size
        y2 = torch.sigmoid(raw[:, 8:9, :]) * self.img_size
        x3 = torch.sigmoid(raw[:, 9:10, :]) * self.img_size
        y3 = torch.sigmoid(raw[:, 10:11, :]) * self.img_size
        x4 = torch.sigmoid(raw[:, 11:12, :]) * self.img_size
        y4 = torch.sigmoid(raw[:, 12:13, :]) * self.img_size

        vis = torch.full((b, 1, raw.shape[2]), 2.0, dtype=x.dtype, device=x.device)

        out = torch.cat(
            [cx, cy, bw, bh, conf, x1, y1, vis, x2, y2, vis, x3, y3, vis, x4, y4, vis],
            dim=1,
        )
        return out


class CardOCRNet(nn.Module):
    """High-accuracy CRNN / Temporal ConvNet with 128 time steps for card ROI OCR."""

    def __init__(self, num_classes: int = BLANK_IDX + 1) -> None:
        super().__init__()
        # Input: (B, 1, 32, 256) -> Output width: 128 steps (stride 2 horizontally)
        self.cnn = nn.Sequential(
            # 32x256 -> 16x128
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            # 16x128 -> 8x128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # 8x128 -> 4x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # 4x128 -> 2x128
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            # 2x128 -> 1x128
            nn.Conv2d(192, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.seq = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # (B, 256, 1, 128)
        feat = feat.squeeze(2)  # (B, 256, 128)
        logits = self.seq(feat)  # (B, 40, 128)
        return logits.permute(0, 2, 1)  # (B, 128, 40)


def letterbox_sample(
    image: np.ndarray, corners: np.ndarray, model_size: int = 320
) -> tuple[np.ndarray, np.ndarray]:
    """Letterbox image and transform 4 corners to model_size x model_size space."""
    h_orig, w_orig = image.shape[:2]
    scale = min(model_size / w_orig, model_size / h_orig)
    nw, nh = int(w_orig * scale), int(h_orig * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
    pad_x = (model_size - nw) // 2
    pad_y = (model_size - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    ordered = order_points(corners).copy()
    ordered[:, 0] = ordered[:, 0] * scale + pad_x
    ordered[:, 1] = ordered[:, 1] * scale + pad_y
    return blob, ordered


def preprocess_roi_for_ocr(roi: np.ndarray, target_h: int = 32, target_w: int = 256) -> np.ndarray:
    """Convert BGR ROI crop to normalized (1, target_h, target_w) grayscale float32."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return np.expand_dims(norm, axis=0)


def encode_text(text: str) -> list[int]:
    """Encode string into list of character indices in CHAR_MAP."""
    return [CHAR_MAP.index(c) for c in text if c in CHAR_MAP]


def build_training_datasets(
    num_cards: int = 550,
    model_size: int = 320,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    list[tuple[np.ndarray, list[int], str]],
]:
    """Generate both detector and OCR training datasets in a single fast vectorized pass."""
    t0 = time.time()
    det_blobs: list[np.ndarray] = []
    det_targets: list[np.ndarray] = []
    det_has_card: list[float] = []
    ocr_samples: list[tuple[np.ndarray, list[int], str]] = []

    extra_names = [f"{f} {surname}" for f in SAMPLE_FIRSTNAMES for surname in SAMPLE_SURNAMES[:3]]

    for i in range(num_cards):
        cfg = BIN_CONFIGS[i % len(BIN_CONFIGS)]
        pan = generate_luhn_pan(cfg["prefix"], cfg["length"])
        if len(pan) == 15:
            formatted_pan = f"{pan[:4]} {pan[4:10]} {pan[10:]}"
        else:
            formatted_pan = f"{pan[:4]} {pan[4:8]} {pan[8:12]} {pan[12:]}"

        # Cover all months 01..12 and years 25..33 uniformly
        month = (i % 12) + 1
        year = 25 + ((i // 12) % 9)
        expiry = f"{month:02d}/{year:02d}"

        if i % 4 != 0:
            name = SAMPLE_NAMES[i % len(SAMPLE_NAMES)]
        else:
            name = extra_names[i % len(extra_names)]

        card_img = create_base_card_texture(CARD_W, CARD_H)
        draw_card_details(card_img, pan, expiry, name, cfg["network"], cfg["card_class"])

        comp_img, corners = apply_perspective_distortion(card_img, 1280, 960)
        blob, lbox_corners = letterbox_sample(comp_img, corners, model_size=model_size)

        xmin = float(np.min(lbox_corners[:, 0]))
        xmax = float(np.max(lbox_corners[:, 0]))
        ymin = float(np.min(lbox_corners[:, 1]))
        ymax = float(np.max(lbox_corners[:, 1]))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin

        target_vec = np.array(
            [
                cx,
                cy,
                bw,
                bh,
                lbox_corners[0, 0],
                lbox_corners[0, 1],
                lbox_corners[1, 0],
                lbox_corners[1, 1],
                lbox_corners[2, 0],
                lbox_corners[2, 1],
                lbox_corners[3, 0],
                lbox_corners[3, 1],
            ],
            dtype=np.float32,
        )
        det_blobs.append(blob)
        det_targets.append(target_vec)
        det_has_card.append(1.0)

        # Extract rectified ROIs (both exact and slightly perturbed corners for shift invariance)
        for jitter_scale in (0.0, 2.5):
            if jitter_scale > 0.0:
                pert = np.random.uniform(-jitter_scale, jitter_scale, corners.shape).astype(
                    np.float32
                )
                warped = warp_perspective_id1(comp_img, corners + pert)
            else:
                warped = warp_perspective_id1(comp_img, corners)

            rois = get_standard_rois(warped)
            ocr_samples.append(
                (preprocess_roi_for_ocr(rois["pan"]), encode_text(formatted_pan), formatted_pan)
            )
            ocr_samples.append(
                (preprocess_roi_for_ocr(rois["expiry"]), encode_text(expiry), expiry)
            )
            ocr_samples.append((preprocess_roi_for_ocr(rois["name"]), encode_text(name), name))

    # Add 40 negative background-only samples for detector false-positive suppression
    for _ in range(40):
        bg_color = [random.randint(40, 220) for _ in range(3)]
        canvas = np.full((model_size, model_size, 3), bg_color, dtype=np.uint8)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        det_blobs.append(blob)
        det_targets.append(np.zeros(12, dtype=np.float32))
        det_has_card.append(0.0)

    print(
        f"[+] Generated {len(det_blobs)} detector scenes and {len(ocr_samples)} OCR ROI strips in {time.time() - t0:.2f}s"
    )
    det_dataset = (
        torch.tensor(np.stack(det_blobs), dtype=torch.float32),
        torch.tensor(np.stack(det_targets), dtype=torch.float32),
        torch.tensor(np.array(det_has_card, dtype=np.float32), dtype=torch.float32),
    )
    random.shuffle(ocr_samples)
    return det_dataset, ocr_samples


def train_detector(
    det_dataset: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int = 28,
    batch_size: int = 32,
    save_path: Path = PROJECT_ROOT / "card_vision" / "weights" / "card_detector.onnx",
) -> None:
    """Train CardDetectorNet on pre-generated dataset and export to ONNX."""
    all_blobs, all_targets, all_has_card = det_dataset
    n_samples = all_blobs.shape[0]
    print(f"[*] Training CardDetectorNet on {device} for {epochs} epochs ({n_samples} scenes)...")

    model = CardDetectorNet(img_size=320, grid_size=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        last_corner_err = 0.0
        last_loss = 0.0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            bs = len(idx)
            blobs = all_blobs[idx].to(device)
            targets = all_targets[idx].to(device)
            has_card = all_has_card[idx].to(device)

            preds = model(blobs)  # (B, 17, 100)
            cx_gt = targets[:, 0]
            cy_gt = targets[:, 1]
            gx = torch.clamp((cx_gt / 32.0).long(), 0, 9)
            gy = torch.clamp((cy_gt / 32.0).long(), 0, 9)
            cell_idx = gy * 10 + gx

            conf_target = torch.zeros((bs, 100), dtype=torch.float32, device=device)
            for b_i in range(bs):
                if has_card[b_i] > 0.5:
                    conf_target[b_i, cell_idx[b_i]] = 1.0

            pred_conf = preds[:, 4, :]
            loss_conf = functional.binary_cross_entropy(pred_conf, conf_target)

            pos_mask = has_card > 0.5
            if pos_mask.any():
                b_indices = torch.arange(bs, device=device)[pos_mask]
                c_indices = cell_idx[pos_mask]
                pos_preds = preds[b_indices, :, c_indices]
                pos_coords = torch.cat(
                    [
                        pos_preds[:, 0:4],
                        pos_preds[:, 5:7],
                        pos_preds[:, 8:10],
                        pos_preds[:, 11:13],
                        pos_preds[:, 14:16],
                    ],
                    dim=1,
                )
                loss_reg = functional.smooth_l1_loss(
                    pos_coords / 320.0, targets[pos_mask] / 320.0, beta=0.02
                )
                last_corner_err = float(
                    torch.abs(pos_coords[:, 4:] - targets[pos_mask][:, 4:]).mean().item()
                )
            else:
                loss_reg = torch.tensor(0.0, device=device)

            loss = loss_conf + 12.0 * loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        scheduler.step()
        if epoch % 7 == 0 or epoch == epochs:
            print(
                f"    Epoch {epoch:2d}/{epochs} | Loss: {last_loss:.4f} "
                f"| Mean Corner Error: {last_corner_err:.2f} px ({time.time() - t0:.1f}s)"
            )

    model.eval().cpu()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_in = torch.randn(1, 3, 320, 320)
    torch.onnx.export(
        model,
        dummy_in,
        str(save_path),
        input_names=["images"],
        output_names=["output0"],
        opset_version=14,
        dynamo=False,
    )
    print(f"[+] Exported CardDetectorNet to {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")


def train_ocr(
    samples: list[tuple[np.ndarray, list[int], str]],
    device: torch.device,
    epochs: int = 28,
    batch_size: int = 64,
    save_path: Path = PROJECT_ROOT / "card_vision" / "weights" / "card_ocr.onnx",
) -> None:
    """Train CardOCRNet with CTCLoss and export to ONNX."""
    split = int(len(samples) * 0.92)
    train_samples = samples[:split]
    val_samples = samples[split:]

    print(
        f"[*] Training CardOCRNet on {device} for {epochs} epochs "
        f"({len(train_samples)} train, {len(val_samples)} val)..."
    )
    model = CardOCRNet(num_classes=BLANK_IDX + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_samples)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_samples), batch_size):
            batch = train_samples[i : i + batch_size]
            imgs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
            target_seqs = [b[1] for b in batch]
            target_lengths = torch.tensor([len(seq) for seq in target_seqs], dtype=torch.long)
            targets_flat = torch.tensor(
                [idx for seq in target_seqs for idx in seq], dtype=torch.long
            )

            logits = model(imgs)  # (B, 128, 40)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2).cpu()  # (128, B, 40)
            input_lengths = torch.full((len(batch),), logits.shape[1], dtype=torch.long)

            loss = ctc_loss_fn(log_probs, targets_flat, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        scheduler.step()

        if epoch % 7 == 0 or epoch == epochs:
            model.eval()
            exact = 0
            with torch.no_grad():
                val_imgs = torch.tensor(
                    np.stack([v[0] for v in val_samples]), dtype=torch.float32, device=device
                )
                val_outs = model(val_imgs).cpu().numpy()
                for b_idx, (_, _, gt_str) in enumerate(val_samples):
                    preds = np.argmax(val_outs[b_idx], axis=-1)
                    decoded = []
                    prev = -1
                    for idx in preds:
                        if idx != prev and idx < len(CHAR_MAP):
                            decoded.append(CHAR_MAP[idx])
                        prev = int(idx)
                    if "".join(decoded).strip() == gt_str.strip():
                        exact += 1
            val_acc = (exact / max(1, len(val_samples))) * 100.0
            print(
                f"    Epoch {epoch:2d}/{epochs} | CTC Loss: {total_loss / max(1, num_batches):.4f} "
                f"| Val Exact-Match Acc: {val_acc:.1f}% ({time.time() - t0:.1f}s)"
            )

    model.eval().cpu()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_in = torch.randn(1, 1, 32, 256)
    torch.onnx.export(
        model,
        dummy_in,
        str(save_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamo=False,
    )
    print(f"[+] Exported CardOCRNet to {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")


def evaluate_end_to_end(num_test_cards: int = 25) -> None:
    """Evaluate the exported ONNX models end-to-end via CardScanner."""
    print(f"\n[*] Running End-to-End Evaluation on {num_test_cards} perspective-warped cards...")
    scanner = CardScanner()
    assert scanner.yolo_session is not None, "Detector ONNX session failed to load!"
    assert scanner.recognizer.session is not None, "OCR ONNX session failed to load!"

    pan_matches = 0
    expiry_matches = 0
    name_matches = 0
    network_matches = 0
    luhn_valid_count = 0

    for i in range(num_test_cards):
        cfg = BIN_CONFIGS[i % len(BIN_CONFIGS)]
        pan = generate_luhn_pan(cfg["prefix"], cfg["length"])
        expiry = generate_random_expiry()
        name = SAMPLE_NAMES[i % len(SAMPLE_NAMES)]

        card_img = create_base_card_texture(CARD_W, CARD_H)
        draw_card_details(card_img, pan, expiry, name, cfg["network"], cfg["card_class"])
        comp_img, _ = apply_perspective_distortion(card_img, 1280, 960)

        res = scanner.scan(comp_img, mask_pan_flag=False)
        if res is not None:
            if res.pan == pan:
                pan_matches += 1
            if res.expiry_date == expiry:
                expiry_matches += 1
            if res.cardholder_name == name:
                name_matches += 1
            if res.network != "Unknown":
                network_matches += 1
            if res.is_luhn_valid:
                luhn_valid_count += 1

    print("====================================================")
    print("End-to-End CardScanner ONNX Pipeline Benchmark")
    print("====================================================")
    print(
        f"PAN Exact Accuracy:        {pan_matches}/{num_test_cards} ({pan_matches / num_test_cards * 100:.1f}%)"
    )
    print(
        f"Luhn Valid Extraction:     {luhn_valid_count}/{num_test_cards} ({luhn_valid_count / num_test_cards * 100:.1f}%)"
    )
    print(
        f"Expiry Exact Accuracy:     {expiry_matches}/{num_test_cards} ({expiry_matches / num_test_cards * 100:.1f}%)"
    )
    print(
        f"Cardholder Name Accuracy:  {name_matches}/{num_test_cards} ({name_matches / num_test_cards * 100:.1f}%)"
    )
    print(
        f"BIN Network Resolution:    {network_matches}/{num_test_cards} ({network_matches / num_test_cards * 100:.1f}%)"
    )
    print("====================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export card-vision ONNX models.")
    parser.add_argument("--detector-epochs", type=int, default=28, help="Detector training epochs")
    parser.add_argument("--ocr-epochs", type=int, default=28, help="OCR training epochs")
    parser.add_argument(
        "--num-cards", type=int, default=550, help="Number of synthetic cards to generate"
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    det_dataset, ocr_samples = build_training_datasets(num_cards=args.num_cards)
    train_detector(det_dataset, device=device, epochs=args.detector_epochs)
    train_ocr(ocr_samples, device=device, epochs=args.ocr_epochs)
    evaluate_end_to_end(num_test_cards=25)


if __name__ == "__main__":
    main()
