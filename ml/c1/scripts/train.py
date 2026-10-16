"""Base model training script for C1 road damage detection on RDD2022."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO base detector on RDD2022 dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default="ml/c1/configs/rdd2022_4class.yaml",
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Base model checkpoint (e.g. yolov8s.pt, yolov8m.pt, yolo11m.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Input image resolution.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (-1 for auto-batch).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id (e.g. '0', '0,1') or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="ml/c1/runs/train",
        help="Directory to save training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rdd2022_base_model",
        help="Experiment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' is not installed. Run: uv add ultralytics")
        return

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        print(f"Error: Dataset configuration file not found at '{data_path}'")
        return

    print("=== Launching Base Model Training ===")
    print(f"Config: {data_path}")
    print(f"Base Weights: {args.model}")
    print(f"Epochs: {args.epochs} | Image Size: {args.imgsz} | Batch Size: {args.batch}")
    print(f"Device: {args.device} | Seed: {args.seed}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        resume=args.resume,
        # Road damage specific augmentations
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        save=True,
        save_period=5,
        plots=True,
    )

    print("\n=== Training Completed ===")
    print(f"Best model weights saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
