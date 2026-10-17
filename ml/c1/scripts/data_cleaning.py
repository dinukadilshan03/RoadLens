"""Data Cleaning and 4-Class Sanitization Pipeline for C1 (RDD2022).

Standard Responsibilities:
1. Filters out conversion-specific Class 4 ("Other corruption"), standardizing strictly
   on the 4 official RDD evaluation classes:
   - Class 0: Longitudinal Crack (D00)
   - Class 1: Transverse Crack (D10)
   - Class 2: Alligator Crack (D20)
   - Class 3: Pothole (D40)
2. Sanitizes bounding boxes:
   - Validates coordinates and dimensions.
   - Clips bounding box coordinates touching or slightly exceeding image borders [0.0, 1.0].
   - Filters out degenerate or sub-threshold noise boxes (width/height < min_box_dim).
3. Flexible storage modes:
   - 'symlink' (default): Zero duplicate disk usage on Unix/macOS.
   - 'index': Generates split .txt manifests directly referencing source images.
   - 'copy': Deep copies files (for external filesystems or OS environments without symlink support).
4. Full provenance tracking:
   - Logs country-wise distributions, valid vs. dropped detections, and empty background frames.
   - Outputs a structured provenance_report.json for research documentation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("c1.data_cleaning")

CLASS_NAMES = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean RDD2022 dataset to 4 core distress classes, sanitize bboxes, and track provenance."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/c1/data/RDD_SPLIT"),
        help="Path to source RDD_SPLIT dataset directory (containing train, val, test splits).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("ml/c1/data/rdd_cleaned"),
        help="Target directory to output cleaned 4-class dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "index", "copy"],
        default="symlink",
        help="Dataset construction mode: 'symlink' (instant, zero disk cost), 'index' (txt manifest), or 'copy'.",
    )
    parser.add_argument(
        "--min-box-dim",
        type=float,
        default=0.002,
        help="Minimum normalized width and height threshold to filter zero or near-zero noise boxes.",
    )
    return parser.parse_args()


def extract_country(stem: str) -> str:
    """Extract country prefix from RDD image filename stem."""
    tokens = stem.split("_")
    if tokens[:2] == ["United", "States"]:
        return "United States"
    return tokens[0]


def sanitize_line(line: str, min_box_dim: float) -> tuple[int, float, float, float, float] | None:
    """Validate, clean, and clip a single YOLO annotation line.
    
    Returns (class_id, xc, yc, w, h) if valid, or None if invalid/discarded.
    """
    parts = line.strip().split()
    if not parts or len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
    except (ValueError, IndexError):
        return None

    # Filter out conversion-specific class 4; retain core classes 0, 1, 2, 3
    if class_id not in {0, 1, 2, 3}:
        return None

    # Center sanity check: center must lie within reasonable image bounds
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None

    # Positive dimensions check
    if width <= 0.0 or height <= 0.0:
        return None

    # Clip bounding box edges to [0.0, 1.0] boundary
    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(1.0, x_center + width / 2.0)
    y2 = min(1.0, y_center + height / 2.0)

    clipped_w = x2 - x1
    clipped_h = y2 - y1

    # Filter degenerated or sub-threshold boxes
    if clipped_w < min_box_dim or clipped_h < min_box_dim:
        return None

    clipped_xc = x1 + clipped_w / 2.0
    clipped_yc = y1 + clipped_h / 2.0

    return class_id, clipped_xc, clipped_yc, clipped_w, clipped_h


def process_split(
    split: str,
    source_dir: Path,
    target_dir: Path,
    mode: str,
    min_box_dim: float,
) -> dict:
    """Process, clean, and record provenance for a single dataset split."""
    src_images = source_dir / split / "images"
    src_labels = source_dir / split / "labels"
    dst_images = target_dir / split / "images"
    dst_labels = target_dir / split / "labels"

    dst_labels.mkdir(parents=True, exist_ok=True)
    if mode != "index":
        dst_images.mkdir(parents=True, exist_ok=True)

    image_paths_for_index: list[str] = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    country_image_counts: Counter[str] = Counter()
    country_class_counts: dict[str, Counter[int]] = defaultdict(Counter)
    original_class_counts: Counter[int] = Counter()
    cleaned_class_counts: Counter[int] = Counter()

    stats = {
        "images": 0,
        "valid_boxes": 0,
        "dropped_class4_boxes": 0,
        "invalid_boxes": 0,
        "empty_labels": 0,
        "country_image_counts": country_image_counts,
        "country_class_counts": country_class_counts,
        "original_class_counts": original_class_counts,
        "cleaned_class_counts": cleaned_class_counts,
    }

    image_files = sorted([p for p in src_images.iterdir() if p.suffix.lower() in image_extensions])
    total_images = len(image_files)
    logger.info(f"Split '{split}': Found {total_images:,} images to process.")

    log_interval = max(1000, total_images // 5)

    for idx, img_path in enumerate(image_files, start=1):
        stats["images"] += 1
        country = extract_country(img_path.stem)
        country_image_counts[country] += 1

        # Handle image placement based on mode
        if mode == "index":
            image_paths_for_index.append(str(img_path.resolve()))
        elif mode == "symlink":
            dst_img_path = dst_images / img_path.name
            if not dst_img_path.exists():
                try:
                    os.symlink(img_path.resolve(), dst_img_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst_img_path)
        elif mode == "copy":
            dst_img_path = dst_images / img_path.name
            if not dst_img_path.exists():
                shutil.copy2(img_path, dst_img_path)

        # Process labels
        label_file = src_labels / f"{img_path.stem}.txt"
        dst_label_file = dst_labels / f"{img_path.stem}.txt"

        kept_lines = []
        if label_file.exists():
            raw_lines = label_file.read_text(encoding="utf-8").splitlines()
            for line in raw_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                    original_class_counts[cls_id] += 1
                    if cls_id == 4:
                        stats["dropped_class4_boxes"] += 1
                        continue
                except ValueError:
                    stats["invalid_boxes"] += 1
                    continue

                parsed = sanitize_line(line, min_box_dim)
                if parsed is not None:
                    cid, xc, yc, w, h = parsed
                    kept_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    cleaned_class_counts[cid] += 1
                    country_class_counts[country][cid] += 1
                    stats["valid_boxes"] += 1
                else:
                    stats["invalid_boxes"] += 1

        if not kept_lines:
            stats["empty_labels"] += 1

        dst_label_file.write_text("".join(kept_lines), encoding="utf-8")

        if idx % log_interval == 0 or idx == total_images:
            logger.info(f"Split '{split}': Processed {idx:,}/{total_images:,} images ({idx / total_images * 100:.1f}%)")

    if mode == "index":
        index_file = target_dir / f"{split}.txt"
        index_file.write_text("\n".join(image_paths_for_index) + "\n", encoding="utf-8")

    return stats


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()

    if not source_dir.is_dir():
        logger.error(f"Source directory '{source_dir}' does not exist.")
        return

    logger.info("=== Starting C1 Data Cleaning & 4-Class Sanitization ===")
    logger.info(f"Source Directory : {source_dir}")
    logger.info(f"Target Directory : {target_dir}")
    logger.info(f"Storage Mode     : {args.mode}")
    logger.info(f"Min Box Dim      : {args.min_box_dim}")

    target_dir.mkdir(parents=True, exist_ok=True)
    summary_report = {}

    for split in ["train", "val", "test"]:
        if (source_dir / split).is_dir():
            logger.info(f"--- Processing Split: '{split}' ---")
            stats = process_split(split, source_dir, target_dir, args.mode, args.min_box_dim)
            logger.info(f"Split '{split}' completed:")
            logger.info(f"  Images processed       : {stats['images']:,}")
            logger.info(f"  Valid 4-class boxes    : {stats['valid_boxes']:,}")
            logger.info(f"  Dropped Class 4 boxes  : {stats['dropped_class4_boxes']:,}")
            logger.info(f"  Filtered invalid boxes : {stats['invalid_boxes']:,}")
            logger.info(f"  Empty label frames     : {stats['empty_labels']:,}")

            summary_report[split] = {
                "images": stats["images"],
                "valid_boxes": stats["valid_boxes"],
                "dropped_class4_boxes": stats["dropped_class4_boxes"],
                "invalid_boxes": stats["invalid_boxes"],
                "empty_labels": stats["empty_labels"],
                "original_classes": dict(stats["original_class_counts"]),
                "cleaned_classes": {
                    CLASS_NAMES[k]: v for k, v in stats["cleaned_class_counts"].items()
                },
                "country_images": dict(stats["country_image_counts"]),
                "country_classes": {
                    country: {CLASS_NAMES[k]: v for k, v in c_counts.items()}
                    for country, c_counts in stats["country_class_counts"].items()
                },
            }

    # Save complete provenance report
    report_path = target_dir / "provenance_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    logger.info(f"Provenance report saved to: {report_path}")
    logger.info("=== Data Cleaning Complete ===")


if __name__ == "__main__":
    main()
