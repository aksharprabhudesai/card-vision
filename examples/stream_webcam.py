"""Live Webcam Card Scanning with Real-time HUD and Temporal Consensus."""

import argparse
import sys

import cv2
import numpy as np

from card_vision import StreamCardScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Live camera stream card scanner.")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device index")
    parser.add_argument(
        "--no-mask", dest="mask", action="store_false", help="Do not mask PAN on HUD"
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    scanner = StreamCardScanner(history_size=5, min_consistent_frames=3)
    print("Press 'q' in the camera window to quit, or 'r' to reset temporal tracking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, is_stable = scanner.process_frame(frame, mask_pan_flag=args.mask)
        display = frame.copy()

        # Draw HUD overlays
        if result and result.corners:
            pts = np.array([[int(p.x), int(p.y)] for p in result.corners], dtype=np.int32)
            color = (0, 220, 0) if is_stable else (0, 165, 255)
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=3)

            status_text = "VERIFIED (Luhn Valid)" if is_stable else "SCANNING..."
            cv2.putText(display, status_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            display_pan = result.masked_pan if args.mask else result.pan
            if display_pan:
                cv2.putText(
                    display,
                    f"Card: {display_pan}",
                    (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            if result.expiry_date:
                cv2.putText(
                    display,
                    f"Expires: {result.expiry_date}",
                    (25, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            if result.cardholder_name:
                cv2.putText(
                    display,
                    f"Name: {result.cardholder_name}",
                    (25, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            if result.network.value != "Unknown":
                cv2.putText(
                    display,
                    f"Scheme: {result.network.value} ({result.card_class.value})",
                    (25, 195),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 255, 200),
                    2,
                )
        else:
            cv2.putText(
                display,
                "Align card in camera view",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (120, 120, 120),
                2,
            )

        cv2.imshow("card-vision Live Camera", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            scanner.state.reset()
            print("Stream history reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
