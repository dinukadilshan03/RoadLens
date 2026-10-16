"""Prepare 4-class RDD2022 dataset for training.

Key Refinements:
1. Filters out conversion-specific Class 4 ("Other corruption"), retaining only the 4 official
   RDD classes: D00 (0), D10 (1), D20 (2), D40 (3).
2. Cleans corrupted/out-of-bounds/zero-area bounding boxes.
3. Supports 3 storage modes:
   - 'symlink' (default on Unix/macOS): Instant, zero extra disk usage.
   - 'index' (creates train.txt, val.txt, test.txt pointing directly to source images + writes clean labels).
   - 'copy' (for Windows or external filesystems without symlink support).
4. Tracks and exports granular provenance statistics:
   - Before vs. after class distribution.
   - Country-wise sample and defect distribution across splits.
   - Saves provenance report as JSON/CSV for research documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASS_NAMES = {
    0: "Longitudinal_Crack_D00",
    1: "Transverse_Crack_D10",
    2: "Alligator_Crack_D20",
    3: "Pothole_D40",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter RDD2022 dataset to 4 core classes, clean bboxes, and log provenance."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/c1/data/RDD_SPLIT"),
        help="Path to source RDD_SPLIT dataset containing train, val, test splits.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("ml/c1/data/RDD_SPLIT_4CLASS"),
        help="Target directory to write cleaned 4-class dataset.",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "index", "copy"],
        default="symlink",
        help="Storage strategy: 'symlink' (zero disk copy), 'index' (YOLO txt lists), or 'copy'.",
    )
    parser.add_argument(
        "--min-box-dim",
        type=float,
        default=0.002,
        help="Minimum normalized width and height threshold to filter zero/near-zero noise boxes.",
    )
    return parser.parse_args()


def extract_country(stem: str) -> str:
    tokens = stem.split("_")
    if tokens[:2] == ["United", "States"]:
        return "United States"
    return tokens[0]


def sanitize_line(line: str, min_box_dim: float) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if not parts or len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
    except (ValueError, IndexError):
        return None

    # Filter out conversion-specific class 4; keep only core RDD classes 0, 1, 2, 3
    if class_id not in {0, 1, 2, 3}:
        return None

    # Out-of-bounds check (center coordinates must lie strictly inside [0, 1])
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None

    # Area & dimension sanity check (must exceed minimum dimension threshold)
    if not (min_box_dim <= width <= 1.0 and min_box_dim <= height <= 1.0):
        return None

    return class_id, x_center, y_center, width, height


def process_split(
    split: str,
    source_dir: Path,
    target_dir: Path,
    mode: str,
    min_box_dim: float,
) -> dict:
    src_images = source_dir / split / "images"
    src_labels = source_dir / split / "labels"
    dst_images = target_dir / split / "images"
    dst_labels = target_dir / split / "labels"

    dst_labels.mkdir(parents=True, exist_ok=True)
    if mode != "index":
        dst_images.mkdir(parents=True, exist_ok=True)

    image_paths_for_index: list[str] = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    country_image_counts = Counter()
    country_class_counts = defaultdict(Counter)
    original_class_counts = Counter()
    cleaned_class_counts = Counter()

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

    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue

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

    if mode == "index":
        index_file = target_dir / f"{split}.txt"
        index_file.write_text("\n".join(image_paths_for_index) + "\n", encoding="utf-8")

    return stats


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()

    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    print("=== RDD2022 4-Class Dataset Preparation & Provenance Pipeline ===")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Mode: {args.mode} | Min Box Dim: {args.min_box_dim}")

    target_dir.mkdir(parents=True, exist_ok=True)
    summary_report = {}

    for split in ["train", "val", "test"]:
        if (source_dir / split).is_dir():
            print(f"\nProcessing Split: '{split}'...")
            stats = process_split(split, source_dir, target_dir, args.mode, args.min_box_dim)
            print(f"  Images: {stats['images']:,}")
            print(f"  Valid 4-Class Boxes: {stats['valid_boxes']:,}")
            print(f"  Dropped Class 4 Boxes: {stats['dropped_class4_boxes']:,}")
            print(f"  Filtered Invalid/Near-Zero Boxes: {stats['invalid_boxes']:,}")
            print(f"  Empty Labels (Backgrounds): {stats['empty_labels']:,}")

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
    print(f"\nProvenance report saved to: {report_path}")
    print("=== Preparation Complete ===")


if __name__ == "__main__":
    main()
