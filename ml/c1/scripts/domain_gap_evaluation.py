"""Dual-Domain Evaluation & Gap Analysis Pipeline for C1 Base Model.

Standard Responsibilities:
1. Evaluates base detector across:
   - In-domain test set: RDD2022 international test split.
   - Cross-domain test set: Sri Lankan road video test dataset.
2. Quantifies domain gap degradation:
   - Overall Delta mAP@50 and Delta mAP@50:95.
   - Per-class degradation (D00, D10, D20, D40).
3. Standardized metrics calculation: mAP50, mAP50-95, Precision, Recall, F1-score.
4. Export reports:
   - CSV summary table (domain_gap_summary.csv) for dissertation and presentation graphs.
   - Complete JSON metrics report (domain_gap_report.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("c1.domain_gap_evaluation")

CLASS_NAMES = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}


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
        description="Dual-domain evaluation and cross-domain gap benchmark for C1 detector."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ml/c1/weights/base_model_b0.pt",
        help="Path to trained base model checkpoint.",
    )
    parser.add_argument(
        "--base-data",
        "--rdd-data",
        dest="base_data",
        type=str,
        default="ml/c1/configs/base_model_b0.yaml",
        help="Path to base dataset YAML config (e.g. ml/c1/configs/base_model_b0.yaml).",
    )
    parser.add_argument(
        "--srilanka-data",
        type=str,
        default="ml/c1/configs/srilanka_test.yaml",
        help="Path to Sri Lankan test dataset YAML config.",
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
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="Non-Maximum Suppression (NMS) IoU threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to evaluate on: 'auto', '0', 'cpu', or 'mps'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/c1/runs/domain_gap_eval",
        help="Directory to save evaluation reports and plots.",
    )
    return parser.parse_args()


def extract_metrics(val_results) -> dict:
    """Extract overall and per-class metrics from Ultralytics validation object."""
    metrics = {
        "Overall": {
            "mAP50": float(val_results.box.map50),
            "mAP50-95": float(val_results.box.map),
            "Precision": float(val_results.box.mp),
            "Recall": float(val_results.box.mr),
            "F1": (
                2 * float(val_results.box.mp) * float(val_results.box.mr) /
                (float(val_results.box.mp) + float(val_results.box.mr) + 1e-8)
            ),
        },
        "Per-Class": {},
    }

    ap50_per_class = val_results.box.ap50
    ap_per_class = val_results.box.ap
    p_per_class = val_results.box.p
    r_per_class = val_results.box.r

    for cid, cname in CLASS_NAMES.items():
        if cid < len(ap50_per_class):
            p = float(p_per_class[cid]) if len(p_per_class) > cid else 0.0
            r = float(r_per_class[cid]) if len(r_per_class) > cid else 0.0
            f1 = (2 * p * r) / (p + r + 1e-8)
            metrics["Per-Class"][cname] = {
                "AP50": float(ap50_per_class[cid]),
                "AP50-95": float(ap_per_class[cid]),
                "Precision": p,
                "Recall": r,
                "F1": f1,
            }

    return metrics


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("'ultralytics' is not installed in the environment. Run: uv add ultralytics")
        return

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        # Check standard fallback locations
        fallbacks = [
            Path("ml/c1/weights/base_model_b0.pt").resolve(),
            Path("ml/c1/weights/b0.pt").resolve(),
            Path("ml/c1/weights/base_model_rdd2022.pt").resolve(),
            Path("ml/c1/weights/b0_base_rdd2022.pt").resolve(),
        ]
        found = next((f for f in fallbacks if f.is_file()), None)
        if found:
            weights_path = found
        else:
            logger.error(f"Model weights not found at '{weights_path}'")
            return

    device = auto_detect_device() if args.device == "auto" else args.device

    logger.info("=== Starting C1 Dual-Domain Benchmark Evaluation ===")
    logger.info(f"Weights     : {weights_path}")
    logger.info(f"Resolution  : {args.imgsz}x{args.imgsz}")
    logger.info(f"Thresholds  : IoU={args.iou}, Conf={args.conf}")
    logger.info(f"Device      : {device}")

    model = YOLO(str(weights_path))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_summary = {}

    # 1. In-Domain Evaluation (Base Dataset Test)
    base_yaml = Path(args.base_data).resolve()
    if not base_yaml.is_file():
        legacy_base = Path("ml/c1/configs/rdd2022_4class.yaml").resolve()
        if legacy_base.is_file():
            base_yaml = legacy_base

    if base_yaml.is_file():
        logger.info("--- Step 1: Evaluating In-Domain (Base Dataset Test Split) ---")
        base_val = model.val(
            data=str(base_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            save_json=True,
            plots=True,
            project=str(output_dir),
            name="rdd_test_eval",
        )
        domain_summary["RDD2022_InDomain"] = extract_metrics(rdd_val)
    else:
        logger.warning(f"RDD2022 config not found at '{rdd_yaml}'. Skipping in-domain eval.")

    # 2. Target Domain Evaluation (Sri Lankan Test Set)
    sl_yaml = Path(args.srilanka_data).resolve()
    if sl_yaml.is_file():
        logger.info("--- Step 2: Evaluating Target Domain (Sri Lankan Roads Test Set) ---")
        sl_val = model.val(
            data=str(sl_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            save_json=True,
            plots=True,
            project=str(output_dir),
            name="srilanka_test_eval",
        )
        domain_summary["SriLanka_TargetDomain"] = extract_metrics(sl_val)
    else:
        logger.warning(f"Sri Lanka config not found at '{sl_yaml}'. Skipping target domain eval.")

    # 3. Generate Comparative Report & Domain Gap
    if "RDD2022_InDomain" in domain_summary and "SriLanka_TargetDomain" in domain_summary:
        rdd_m = domain_summary["RDD2022_InDomain"]["Overall"]
        sl_m = domain_summary["SriLanka_TargetDomain"]["Overall"]

        gap_map50 = ((rdd_m["mAP50"] - sl_m["mAP50"]) / rdd_m["mAP50"]) * 100 if rdd_m["mAP50"] > 0 else 0.0
        gap_map = ((rdd_m["mAP50-95"] - sl_m["mAP50-95"]) / rdd_m["mAP50-95"]) * 100 if rdd_m["mAP50-95"] > 0 else 0.0

        logger.info("============================================================")
        logger.info("           DOMAIN GAP BENCHMARK SUMMARY REPORT              ")
        logger.info("============================================================")
        logger.info(f"International mAP@50     : {rdd_m['mAP50'] * 100:.2f}%")
        logger.info(f"Sri Lankan mAP@50        : {sl_m['mAP50'] * 100:.2f}%")
        logger.info(f"-> Relative Drop (ΔmAP@50)   : {gap_map50:.2f}%")
        logger.info(f"-> Relative Drop (ΔmAP@50:95): {gap_map:.2f}%")

        rows = []
        for cname in CLASS_NAMES.values():
            rdd_c = domain_summary["RDD2022_InDomain"]["Per-Class"].get(cname, {})
            sl_c = domain_summary["SriLanka_TargetDomain"]["Per-Class"].get(cname, {})

            rdd_ap = rdd_c.get("AP50", 0.0)
            sl_ap = sl_c.get("AP50", 0.0)
            c_gap = ((rdd_ap - sl_ap) / rdd_ap) * 100 if rdd_ap > 0 else 0.0

            rows.append({
                "Damage Class": cname,
                "RDD AP@50": round(rdd_ap * 100, 2),
                "Sri Lanka AP@50": round(sl_ap * 100, 2),
                "Drop (ΔAP@50 %)": round(c_gap, 2),
                "RDD F1": round(rdd_c.get("F1", 0.0), 3),
                "Sri Lanka F1": round(sl_c.get("F1", 0.0), 3),
            })

        try:
            import pandas as pd
            df_classes = pd.DataFrame(rows)
            logger.info("\n--- Per-Class Degradation Table ---")
            logger.info("\n" + df_classes.to_string(index=False))

            csv_path = output_dir / "domain_gap_summary.csv"
            df_classes.to_csv(csv_path, index=False)
            logger.info(f"Per-class domain gap saved to: {csv_path}")
        except ImportError:
            import csv
            csv_path = output_dir / "domain_gap_summary.csv"
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            logger.info(f"Per-class domain gap saved to: {csv_path}")

    report_json = output_dir / "domain_gap_report.json"
    report_json.write_text(json.dumps(domain_summary, indent=2), encoding="utf-8")
    logger.info(f"Complete evaluation report JSON saved to: {report_json}")
    logger.info("=== Domain Gap Evaluation Complete ===")


if __name__ == "__main__":
    main()
