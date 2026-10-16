"""Base model training script for C1 road damage detection on RDD2022.

Refinements:
1. Full reproducibility: seed, deterministic execution, and PyTorch backend enforcement.
2. YAML hyperparameter decoupling (reads configs/hyp_base.yaml or CLI override).
3. Automatic checkpoint management:
   - Saves run logs & curves in ml/c1/runs/train/<name>/
   - Copies best checkpoint directly to ml/c1/weights/b0_base_rdd2022.pt for evaluation.
4. Per-epoch training & validation progress tracking with early stopping patience.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO base detector (B0) on 4-Class RDD2022 dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ml/c1/configs/rdd2022_4class.yaml",
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default="ml/c1/configs/hyp_base.yaml",
        help="Path to hyperparameters YAML config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="Base model weights (e.g. yolov8s.pt, yolov8m.pt, yolo11m.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement).",
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
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker threads.",
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
        default="b0_base_rdd2022",
        help="Experiment name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reproducibility.",
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
    hyp_path = Path(args.hyp).resolve() if args.hyp else None

    if not data_path.is_file():
        print(f"Error: Dataset configuration file not found at '{data_path}'")
        return

    print("=== Launching C1 Base Model Training (B0 Baseline) ===")
    print(f"Dataset Config : {data_path}")
    print(f"Hyp Config     : {hyp_path if hyp_path and hyp_path.is_file() else 'Ultralytics default'}")
    print(f"Base Weights   : {args.model}")
    print(f"Epochs         : {args.epochs} (Patience: {args.patience})")
    print(f"Resolution     : {args.imgsz}x{args.imgsz} | Batch Size: {args.batch}")
    print(f"Device         : {args.device} | Seed: {args.seed} | Deterministic: True")

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "deterministic": True,
        "resume": args.resume,
        "save": True,
        "save_period": 5,
        "plots": True,
    }

    # Pass external hyperparameters file if present
    if hyp_path and hyp_path.is_file():
        train_kwargs["cfg"] = str(hyp_path)

    results = model.train(**train_kwargs)

    # Manage checkpoint persistence to ml/c1/weights/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        weights_dir = Path("ml/c1/weights").resolve()
        weights_dir.mkdir(parents=True, exist_ok=True)
        dest_pt = weights_dir / "b0_base_rdd2022.pt"
        shutil.copy2(best_pt, dest_pt)
        print(f"\nSuccessfully copied best model to: {dest_pt}")

    print("\n=== Training Completed Successfully ===")
    print(f"Training run logs and evaluation plots saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
