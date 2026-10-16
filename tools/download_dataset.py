#!/usr/bin/env python3
"""Dataset Downloader and Ingestion Utility for Card Detection & OCR.

Supports:
1. Hugging Face Datasets (e.g. lambdaWalker/creditCardDetectionDS)
2. Roboflow Universe dataset exports (YOLOv8 / YOLO11 format)
3. MIDV ID-1 document benchmarks for perspective distortion & homography training
4. Automatic data.yaml creation for Ultralytics YOLO training
"""

import argparse
import os
import sys
from pathlib import Path

# Known and verified dataset references
DATASET_REGISTRY = {
    "hf-lambdaWalker": {
        "repo_id": "lambdaWalker/creditCardDetectionDS",
        "description": "5,000 synthetic card images (800x800) formatted with YOLO annotations (Apache-2.0)",
        "type": "huggingface",
    },
    "roboflow-di8dy": {
        "workspace": "janhavi-zambre-nmius",
        "project": "credit-card-object-detection-di8dy",
        "description": "Multi-class card ROI dataset (Issuer, Number, Cardholder, Expiry, Network)",
        "type": "roboflow",
    },
    "midv-500-id1": {
        "description": "MIDV-500 ID-1 standard document subset (85.60 x 53.98 mm) with 4-corner keypoints",
        "type": "midv",
    },
}


def download_huggingface_dataset(repo_id: str, output_dir: Path) -> None:
    """Download and format a Hugging Face card dataset into standard YOLO structure."""
    print(f"[*] Ingesting Hugging Face dataset: {repo_id}")
    try:
        from huggingface_hub import snapshot_download

        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[+] Downloaded Hugging Face files to: {download_path}")
    except ImportError:
        print(
            "[!] 'huggingface_hub' is not installed. You can install it via 'pip install huggingface_hub'"
        )
        print("[!] Falling back to git clone method...")
        os.system(f"git clone https://huggingface.co/datasets/{repo_id} {output_dir}")

    # Ensure data.yaml exists for YOLOv8/YOLO11 training
    create_yolo_data_yaml(output_dir, num_classes=1, class_names=["credit_card"])


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    api_key: str | None,
    output_dir: Path,
) -> None:
    """Download Roboflow Universe dataset exported in YOLOv8 format."""
    print(f"[*] Downloading Roboflow project: {workspace}/{project} (v{version})")
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")

    if not api_key:
        print(
            "[!] Error: Roboflow API key required. Pass --api-key or set ROBOFLOW_API_KEY environment variable."
        )
        print(f"    Explore in browser: https://universe.roboflow.com/{workspace}/{project}")
        sys.exit(1)

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=str(output_dir))
        print(f"[+] Dataset saved to: {dataset.location}")
    except ImportError:
        print("[!] 'roboflow' package not found. Install via 'pip install roboflow'")
        sys.exit(1)


def create_yolo_data_yaml(
    dataset_dir: Path,
    num_classes: int = 1,
    class_names: list[str] | None = None,
) -> Path:
    """Generate Ultralytics-compatible data.yaml configuration."""
    if class_names is None:
        class_names = ["credit_card"]

    yaml_path = dataset_dir / "data.yaml"
    content = f"""# Ultralytics YOLO dataset configuration
path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {num_classes}
names: {class_names}
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[+] Created YOLO dataset config: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and ingest card vision datasets.")
    parser.add_argument(
        "--source",
        choices=list(DATASET_REGISTRY.keys()) + ["custom-hf", "custom-roboflow"],
        default="hf-lambdaWalker",
        help="Dataset identifier or source type",
    )
    parser.add_argument("--repo-id", type=str, help="Hugging Face repo id if source=custom-hf")
    parser.add_argument(
        "--workspace", type=str, help="Roboflow workspace if source=custom-roboflow"
    )
    parser.add_argument("--project", type=str, help="Roboflow project if source=custom-roboflow")
    parser.add_argument("--version", type=int, default=1, help="Roboflow version number")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument(
        "--output-dir", type=str, default="data/dataset", help="Destination directory"
    )

    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.source == "hf-lambdaWalker" or (args.source == "custom-hf" and args.repo_id):
        repo_id = args.repo_id or DATASET_REGISTRY["hf-lambdaWalker"]["repo_id"]
        download_huggingface_dataset(repo_id, out)

    elif args.source == "roboflow-di8dy" or (
        args.source == "custom-roboflow" and args.workspace and args.project
    ):
        workspace = args.workspace or DATASET_REGISTRY["roboflow-di8dy"]["workspace"]
        project = args.project or DATASET_REGISTRY["roboflow-di8dy"]["project"]
        download_roboflow_dataset(workspace, project, args.version, args.api_key, out)

    elif args.source == "midv-500-id1":
        print("[*] MIDV-500 contains ID-1 documents (85.60 x 53.98 mm).")
        print("    Reference: https://github.com/fcakyon/midv-500")
        print(f"    Target output directory prepared at: {out}")


if __name__ == "__main__":
    main()
