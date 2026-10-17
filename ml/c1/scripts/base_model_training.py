"""Base Model Training Pipeline for C1 (Road Damage Detection).

Standard Responsibilities:
1. Trains baseline international detector (YOLOv8m / YOLO11m) on 4-Class RDD2022 dataset.
2. Full reproducibility: enforces deterministic seeding and PyTorch backend settings.
3. Decoupled YAML hyperparameter loading (reads ml/c1/configs/hyp_base.yaml).
4. Automatic hardware detection (CUDA -> Apple Silicon MPS -> CPU).
5. Automated checkpoint persistence:
   - Tracks metrics, curves, and validation plots in ml/c1/runs/train/<name>/
   - Copies best checkpoint to ml/c1/weights/base_model_rdd2022.pt.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("c1.base_model_training")


def auto_detect_device() -> str:
    """Detect available compute acceleration: CUDA -> Apple Silicon MPS -> CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO base detector on 4-Class RDD2022 international dataset."
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
        help="Base model architecture or initial weights (e.g. yolov8s.pt, yolov8m.pt, yolo11m.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without validation improvement).",
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
        default="auto",
        help="Compute device: 'auto' (detects CUDA/MPS/CPU), '0', 'cpu', or 'mps'.",
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
        help="Directory to save training run artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="base_model_rdd2022",
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
        logger.error("'ultralytics' is not installed in the environment. Run: uv add ultralytics")
        return

    data_path = Path(args.data).resolve()
    hyp_path = Path(args.hyp).resolve() if args.hyp else None

    if not data_path.is_file():
        logger.error(f"Dataset configuration file not found at '{data_path}'")
        return

    device = auto_detect_device() if args.device == "auto" else args.device

    logger.info("=== Launching C1 Base Model Training ===")
    logger.info(f"Dataset Config : {data_path}")
    logger.info(f"Hyperparameters: {hyp_path if hyp_path and hyp_path.is_file() else 'Ultralytics default'}")
    logger.info(f"Base Weights   : {args.model}")
    logger.info(f"Epochs         : {args.epochs} (Early Stopping Patience: {args.patience})")
    logger.info(f"Resolution     : {args.imgsz}x{args.imgsz} | Batch Size: {args.batch}")
    logger.info(f"Device         : {device} | Seed: {args.seed} | Deterministic: True")

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
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

    if hyp_path and hyp_path.is_file():
        train_kwargs["cfg"] = str(hyp_path)

    model.train(**train_kwargs)

    # Manage checkpoint persistence to ml/c1/weights/
    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        weights_dir = Path("ml/c1/weights").resolve()
        weights_dir.mkdir(parents=True, exist_ok=True)
        dest_pt = weights_dir / "base_model_rdd2022.pt"
        shutil.copy2(best_pt, dest_pt)
        # Also maintain legacy alias
        shutil.copy2(best_pt, weights_dir / "b0_base_rdd2022.pt")
        logger.info(f"Best model weights successfully saved to: {dest_pt}")

    logger.info("=== Base Model Training Completed Successfully ===")
    logger.info(f"Artifacts and validation plots saved to: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
