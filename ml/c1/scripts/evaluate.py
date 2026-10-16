"""Dual-domain evaluation script for C1.

Refinements & Features:
1. Standardized Metrics:
   - Overall mAP@50, mAP@50:95, Precision, Recall, F1-score.
2. Per-Class Granularity:
   - AP@50, AP@50:95, Precision, Recall for:
     * D00: Longitudinal Crack
     * D10: Transverse Crack
     * D20: Alligator Crack
     * D40: Pothole
3. Country-Slice Breakdown for RDD2022:
   - Evaluates sub-splits for Japan, India, Czech, Norway, US, China to analyze regional domain differences.
4. Domain Gap Calculation:
   - ΔmAP = (mAP_in_domain - mAP_cross_domain) / mAP_in_domain * 100%
5. Exports:
   - Markdown summary table matching Table 4 & Table 5 in proposal.
   - CSV export for dissertation plotting and analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

CLASS_NAMES = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-domain evaluation & domain gap benchmark (B0 baseline)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="ml/c1/weights/b0_base_rdd2022.pt",
        help="Path to trained base model weights.",
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
        default=0.50,
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
        help="Directory to save evaluation reports and CSVs.",
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

    # Per-class AP arrays
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
        print("Error: 'ultralytics' is not installed. Run: uv add ultralytics")
        return

    weights_path = Path(args.weights).resolve()
    if not weights_path.is_file():
        print(f"Error: Model weights not found at '{weights_path}'")
        return

    print("=== Loading Base Model for Dual-Domain Benchmark ===")
    print(f"Weights     : {weights_path}")
    print(f"Resolution  : {args.imgsz}x{args.imgsz}")
    print(f"Thresholds  : IoU={args.iou}, Conf={args.conf}")

    model = YOLO(str(weights_path))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_summary = {}

    # 1. In-Domain Evaluation (RDD2022 Test)
    rdd_yaml = Path(args.rdd_data).resolve()
    if rdd_yaml.is_file():
        print("\n------------------------------------------------------------")
        print("1. Evaluating on In-Domain International Test Set (RDD2022)")
        print("------------------------------------------------------------")
        rdd_val = model.val(
            data=str(rdd_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_json=True,
            plots=True,
            project=str(output_dir),
            name="rdd_test_eval",
        )
        domain_summary["RDD2022_InDomain"] = extract_metrics(rdd_val)

    # 2. Target Cross-Domain Evaluation (Sri Lankan Test Set)
    sl_yaml = Path(args.srilanka_data).resolve()
    if sl_yaml.is_file():
        print("\n------------------------------------------------------------")
        print("2. Evaluating on Target Cross-Domain Test Set (Sri Lanka)")
        print("------------------------------------------------------------")
        sl_val = model.val(
            data=str(sl_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_json=True,
            plots=True,
            project=str(output_dir),
            name="srilanka_test_eval",
        )
        domain_summary["SriLanka_TargetDomain"] = extract_metrics(sl_val)

    # 3. Generate Comparative Report & Domain Gap
    if "RDD2022_InDomain" in domain_summary and "SriLanka_TargetDomain" in domain_summary:
        rdd_m = domain_summary["RDD2022_InDomain"]["Overall"]
        sl_m = domain_summary["SriLanka_TargetDomain"]["Overall"]

        gap_map50 = ((rdd_m["mAP50"] - sl_m["mAP50"]) / rdd_m["mAP50"]) * 100 if rdd_m["mAP50"] > 0 else 0.0
        gap_map = ((rdd_m["mAP50-95"] - sl_m["mAP50-95"]) / rdd_m["mAP50-95"]) * 100 if rdd_m["mAP50-95"] > 0 else 0.0

        print("\n============================================================")
        print("          B0 BASELINE: DOMAIN GAP SUMMARY REPORT            ")
        print("============================================================")
        print(f"International mAP@50   : {rdd_m['mAP50'] * 100:.2f}%")
        print(f"Sri Lankan mAP@50      : {sl_m['mAP50'] * 100:.2f}%")
        print(f"-> Relative Drop (ΔmAP@50) : {gap_map50:.2f}%")
        print(f"-> Relative Drop (ΔmAP@50:95) : {gap_map:.2f}%")

        # Build Per-Class Comparison DataFrame
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

        df_classes = pd.DataFrame(rows)
        print("\n--- Per-Class Performance & Degradation ---")
        print(df_classes.to_string(index=False))

        csv_path = output_dir / "b0_per_class_domain_gap.csv"
        df_classes.to_csv(csv_path, index=False)
        print(f"\nPer-class domain gap saved to: {csv_path}")

    # Save complete JSON report
    report_json = output_dir / "b0_evaluation_report.json"
    report_json.write_text(json.dumps(domain_summary, indent=2), encoding="utf-8")
    print(f"Full evaluation JSON report saved to: {report_json}")


if __name__ == "__main__":
    main()
