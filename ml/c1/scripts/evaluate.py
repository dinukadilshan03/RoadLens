"""Dual-domain evaluation script.

Evaluates the trained base model on:
1. In-domain international test set (RDD2022)
2. Target cross-domain test set (Sri Lankan roads)
Computes the domain gap: ΔmAP = (mAP_in_domain - mAP_cross_domain) / mAP_in_domain * 100%
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate domain gap between RDD2022 and Sri Lankan roads.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained base model weights (e.g. ml/c1/weights/best.pt).",
    )
    parser.add_argument(
        "--rdd-data",
        type=str,
        default="ml/c1/configs/rdd2022_4class.yaml",
        help="Dataset YAML config for RDD2022.",
    )
    parser.add_argument(
        "--srilanka-data",
        type=str,
        default="ml/c1/configs/srilanka_test.yaml",
        help="Dataset YAML config for Sri Lankan test dataset.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Evaluation image resolution.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id or 'cpu'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/c1/runs/domain_eval",
        help="Directory to save evaluation outputs and domain gap report.",
    )
    return parser.parse_args()


def evaluate_domain(model, data_config: str, split: str, imgsz: int, conf: float, iou: float, device: str):
    metrics = model.val(
        data=data_config,
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save_json=True,
        plots=True,
    )
    return {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
    }


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' is not installed. Run: uv add ultralytics")
        return

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        print(f"Error: Model weights not found at '{weights_path}'")
        return

    print("=== Loading Base Model ===")
    print(f"Weights: {weights_path}")
    model = YOLO(str(weights_path))

    results = {}

    # 1. In-domain evaluation (RDD2022 test)
    rdd_path = Path(args.rdd_data).resolve()
    if rdd_path.is_file():
        print("\n--- Evaluating on In-Domain Dataset (RDD2022 Test) ---")
        results["RDD2022 (In-Domain)"] = evaluate_domain(
            model, str(rdd_path), "test", args.imgsz, args.conf, args.iou, args.device
        )
    else:
        print(f"Warning: RDD2022 config not found at '{rdd_path}'. Skipping.")

    # 2. Target domain evaluation (Sri Lanka test)
    sl_path = Path(args.srilanka_data).resolve()
    if sl_path.is_file():
        print("\n--- Evaluating on Target Domain (Sri Lanka Test) ---")
        results["Sri Lanka (Target Domain)"] = evaluate_domain(
            model, str(sl_path), "test", args.imgsz, args.conf, args.iou, args.device
        )
    else:
        print(f"Warning: Sri Lanka config not found at '{sl_path}'. Skipping.")

    # Compute comparison table
    if results:
        df_results = pd.DataFrame(results).T
        if "RDD2022 (In-Domain)" in df_results.index and "Sri Lanka (Target Domain)" in df_results.index:
            rdd_map50 = df_results.loc["RDD2022 (In-Domain)", "mAP50"]
            sl_map50 = df_results.loc["Sri Lanka (Target Domain)", "mAP50"]
            gap = ((rdd_map50 - sl_map50) / rdd_map50) * 100 if rdd_map50 > 0 else 0.0

            print("\n=== Domain Gap Performance Comparison ===")
            print(df_results.round(4))
            print(f"\nRelative mAP@50 Degradation on Sri Lankan Roads: {gap:.2f}%")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "domain_gap_summary.csv"
        df_results.to_csv(csv_path)
        print(f"Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
