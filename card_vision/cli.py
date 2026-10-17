"""CLI entrypoint for card-vision."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from card_vision import __version__
from card_vision.scanner import CardScanner
from card_vision.stream import StreamCardScanner


def run_scan(args: argparse.Namespace) -> None:
    """Execute single image scan."""
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


def run_stream(args: argparse.Namespace) -> None:
    """Execute live video / camera feed card scanning with temporal voting."""
    source: int | str
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.stderr.write(f"Error: Could not open video source '{source}'\n")
        sys.exit(1)

    stream_scanner = StreamCardScanner()
    sys.stdout.write(f"[*] Starting video scan from source {source}. Press 'q' to quit...\n")

    from collections.abc import Iterator

    from card_vision.models import CardResult

    def frame_gen() -> Iterator[np.ndarray]:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    def hud_callback(frame: np.ndarray, res: CardResult | None, stable: bool) -> None:
        if args.headless:
            if stable and res and res.pan:
                sys.stdout.write(f"[+] Stable Card Verified: {res.pan}\n")
            return

        display_frame = frame.copy()
        if res and res.corners:
            pts = np.array([[int(p.x), int(p.y)] for p in res.corners], dtype=np.int32)
            color = (0, 255, 0) if stable else (0, 165, 255)
            cv2.polylines(display_frame, [pts], isClosed=True, color=color, thickness=3)

            # Draw attributes HUD
            status = "STABLE CONSENSUS" if stable else "SEARCHING / SCANNING"
            cv2.putText(
                display_frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            if res.masked_pan:
                cv2.putText(
                    display_frame,
                    f"PAN: {res.masked_pan}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            if res.expiry_date:
                cv2.putText(
                    display_frame,
                    f"EXP: {res.expiry_date}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        cv2.imshow("card-vision Live Scanner", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()

    result = stream_scanner.scan_video_stream(
        frame_generator=frame_gen(),
        on_frame_processed=hud_callback,
        stop_on_stable=not args.continuous,
        mask_pan_flag=args.mask,
    )

    cap.release()
    cv2.destroyAllWindows()

    if result:
        sys.stdout.write("\n=== Confirmed Card Result ===\n")
        sys.stdout.write(f"Network:     {result.network.value}\n")
        sys.stdout.write(f"Card Class:  {result.card_class.value}\n")
        sys.stdout.write(f"PAN:         {result.pan}\n")
        sys.stdout.write(f"Expiry:      {result.expiry_date}\n")
        sys.stdout.write(f"Cardholder:  {result.cardholder_name}\n")
        sys.stdout.write(f"Confidence:  {result.confidence:.2f}\n")


def run_serve(args: argparse.Namespace) -> None:
    """Start FastAPI microservice."""
    try:
        import uvicorn

        from card_vision.server import create_app

        app = create_app()
        sys.stdout.write(f"[*] Starting card-vision server at http://{args.host}:{args.port}\n")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        sys.stderr.write(
            "Error: uvicorn and fastapi are required for server mode.\n"
            "Run: pip install fastapi uvicorn python-multipart\n"
        )
        sys.exit(1)


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        prog="card-vision",
        description="Extract and validate financial card attributes with privacy-first CV/OCR.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: scan
    scan_p = subparsers.add_parser("scan", help="Scan a single image file")
    scan_p.add_argument("--image", "-i", type=str, required=True, help="Path to input card image")
    scan_p.add_argument(
        "--mask", action="store_true", default=True, help="Mask PAN digits in output"
    )
    scan_p.add_argument(
        "--no-mask", dest="mask", action="store_false", help="Expose raw PAN digits"
    )
    scan_p.add_argument("--yolo-model", type=str, default=None, help="Custom YOLO ONNX model path")
    scan_p.add_argument("--ocr-model", type=str, default=None, help="Custom OCR ONNX model path")
    scan_p.add_argument("--json", action="store_true", help="Output result as JSON")

    # Command: stream
    stream_p = subparsers.add_parser("stream", help="Scan live camera feed or video file")
    stream_p.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Camera device index (e.g. 0) or path to video file",
    )
    stream_p.add_argument(
        "--mask", action="store_true", default=True, help="Mask PAN digits in output"
    )
    stream_p.add_argument(
        "--no-mask", dest="mask", action="store_false", help="Expose raw PAN digits"
    )
    stream_p.add_argument(
        "--continuous", action="store_true", help="Keep scanning continuously without exiting"
    )
    stream_p.add_argument("--headless", action="store_true", help="Run without opening GUI window")

    # Command: serve
    serve_p = subparsers.add_parser("serve", help="Start REST API microservice")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to listen on")

    # Command: version
    subparsers.add_parser("version", help="Print card-vision library version")

    args = parser.parse_args()

    if args.command == "scan":
        run_scan(args)
    elif args.command == "stream":
        run_stream(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "version":
        sys.stdout.write(f"card-vision {__version__}\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
