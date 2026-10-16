"""Prepare 4-class RDD2022 dataset for training.

This script:
1. Filters out conversion-specific Class 4 ("Other corruption"), retaining only the 4 official
   RDD classes: D00 (0), D10 (1), D20 (2), D40 (3).
2. Cleans corrupted/out-of-bound/zero-area bounding boxes.
3. Links or copies images and writes sanitized label files to the target 4-class directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter RDD2022 dataset to 4 core classes and sanitize annotations."
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
        "--copy-images",
        action="store_true",
        help="Copy image files instead of symlinking (use on Windows or non-symlink filesystems).",
    )
    return parser.parse_args()


def sanitize_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if not parts or len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
    except (ValueError, IndexError):
        return None

    # Filter out conversion-specific class 4; only keep core RDD classes 0, 1, 2, 3
    if class_id not in {0, 1, 2, 3}:
        return None

    # Filter out invalid or zero-dimension bounding boxes
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return None

    return class_id, x_center, y_center, width, height


def process_split(
    split: str,
    source_dir: Path,
    target_dir: Path,
    copy_images: bool,
) -> dict[str, int]:
    src_images = source_dir / split / "images"
    src_labels = source_dir / split / "labels"
    dst_images = target_dir / split / "images"
    dst_labels = target_dir / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "valid_boxes": 0,
        "dropped_class4_boxes": 0,
        "invalid_boxes": 0,
        "empty_labels": 0,
    }

    image_extensions = {".jpg", ".jpeg", ".png"}

    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        stats["images"] += 1

        # Link or copy image
        dst_img_path = dst_images / img_path.name
        if not dst_img_path.exists():
            if copy_images:
                shutil.copy2(img_path, dst_img_path)
            else:
                try:
                    os.symlink(img_path.resolve(), dst_img_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst_img_path)

        # Process label file
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
                    if cls_id == 4:
                        stats["dropped_class4_boxes"] += 1
                        continue
                except ValueError:
                    stats["invalid_boxes"] += 1
                    continue

                parsed = sanitize_line(line)
                if parsed is not None:
                    kept_lines.append(
                        f"{parsed[0]} {parsed[1]:.6f} {parsed[2]:.6f} {parsed[3]:.6f} {parsed[4]:.6f}\n"
                    )
                    stats["valid_boxes"] += 1
                else:
                    stats["invalid_boxes"] += 1

        if not kept_lines:
            stats["empty_labels"] += 1

        dst_label_file.write_text("".join(kept_lines), encoding="utf-8")

    return stats


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()

    if not source_dir.is_dir():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    print(f"Preparing 4-class RDD dataset...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")

    total_stats: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        if (source_dir / split).is_dir():
            print(f"\nProcessing split: {split}...")
            stats = process_split(split, source_dir, target_dir, args.copy_images)
            print(f"  Images: {stats['images']:,}")
            print(f"  Valid 4-class boxes: {stats['valid_boxes']:,}")
            print(f"  Dropped Class 4 boxes: {stats['dropped_class4_boxes']:,}")
            print(f"  Filtered invalid boxes: {stats['invalid_boxes']:,}")
            print(f"  Empty label files (backgrounds): {stats['empty_labels']:,}")
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

    print("\n=== Dataset Preparation Complete ===")
    print(f"Total images processed: {total_stats.get('images', 0):,}")
    print(f"Total 4-class boxes retained: {total_stats.get('valid_boxes', 0):,}")
    print(f"Total Class 4 boxes dropped: {total_stats.get('dropped_class4_boxes', 0):,}")
    print(f"Total invalid boxes discarded: {total_stats.get('invalid_boxes', 0):,}")


if __name__ == "__main__":
    main()
