"""IRT Pavement Dataset Loader & Tropical Optical Noise Augmentations.

Supports paired loading of RGB road surface crops and co-registered thermal images.
Implements route-based geographic/session partitioning and asymmetric noise injection
(simulated dynamic tree canopy shadows, wet road specular glare) to train the
cGAN generator on noise resilience.

References:
- Proposal Section 3.2 Dataset Specifications (#1 IRT Pavement Dataset)
- Proposal Section 3.6 Route-Based Geographical Partitioning
- Proposal Figure 1.1 Environmental Noise Corruption vs Software Thermal Synthesis
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class AsymmetricOpticalNoise:
    """Injects synthetic tree canopy shadows and solar glare solely onto the RGB input."""

    @staticmethod
    def apply_synthetic_shadow(image: np.ndarray) -> np.ndarray:
        """Simulates dynamic tree canopy shadows across the asphalt surface."""
        h, w = image.shape[:2]
        num_shadows = np.random.randint(1, 4)
        shadow_mask = np.ones((h, w), dtype=np.float32)

        for _ in range(num_shadows):
            num_points = np.random.randint(3, 7)
            points = np.column_stack([
                np.random.randint(0, w, size=num_points),
                np.random.randint(0, h, size=num_points),
            ])
            cv2.fillPoly(shadow_mask, [points], np.random.uniform(0.35, 0.65))

        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        shadowed = (image.astype(np.float32) * shadow_mask[:, :, None]).clip(0, 255).astype(np.uint8)
        return shadowed

    @staticmethod
    def apply_solar_glare(image: np.ndarray) -> np.ndarray:
        """Simulates localized tropical sun glare / wet road specular reflection."""
        h, w = image.shape[:2]
        center_x = np.random.randint(0, w)
        center_y = np.random.randint(0, h)
        radius = np.random.randint(min(h, w) // 6, min(h, w) // 2)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        glare_intensity = np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2)) * np.random.uniform(120, 200)

        glared = image.astype(np.float32) + glare_intensity[:, :, None]
        return np.clip(glared, 0, 255).astype(np.uint8)


def discover_paired_dataset(rgb_dir: Path, thermal_dir: Path) -> List[Tuple[Path, Path]]:
    """Discovers and pairs corresponding RGB and Thermal images by filename stem."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    rgb_files = {p.stem: p for p in rgb_dir.iterdir() if p.suffix.lower() in valid_exts}
    thermal_files = {p.stem: p for p in thermal_dir.iterdir() if p.suffix.lower() in valid_exts}

    common_stems = sorted(list(set(rgb_files.keys()).intersection(set(thermal_files.keys()))))

    if len(common_stems) == 0:
        norm_rgb = {re.sub(r"^(rgb|thermal|t|vis)[_-]?", "", k, flags=re.IGNORECASE): v for k, v in rgb_files.items()}
        norm_thm = {re.sub(r"^(rgb|thermal|t|vis)[_-]?", "", k, flags=re.IGNORECASE): v for k, v in thermal_files.items()}
        common_norm = sorted(list(set(norm_rgb.keys()).intersection(set(norm_thm.keys()))))
        pairs = [(norm_rgb[k], norm_thm[k]) for k in common_norm]
    else:
        pairs = [(rgb_files[s], thermal_files[s]) for s in common_stems]

    return pairs


def split_by_route_session(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Strictly partitions dataset by route/session (date prefix YYYYMMDD) to prevent temporal frame leakage.
    
    References Proposal Section 3.6: 'To prevent optimistic evaluation bias caused by
    temporal frame leakage, datasets are split strictly using Route-Based Geographical Partitioning.'
    """
    # Group pairs by session prefix (e.g. '20221104')
    sessions: Dict[str, List[Tuple[Path, Path]]] = {}
    for rgb_path, thm_path in pairs:
        prefix = rgb_path.stem[:8] if len(rgb_path.stem) >= 8 and rgb_path.stem[:8].isdigit() else "default"
        sessions.setdefault(prefix, []).append((rgb_path, thm_path))

    if len(sessions) <= 1:
        # Fallback to deterministic pseudo-random split if no session prefixes exist
        rng = random.Random(seed)
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        split = int((1.0 - val_ratio) * len(shuffled))
        return shuffled[:split], shuffled[split:]

    # Sort session keys for determinism
    sorted_sessions = sorted(sessions.keys())
    rng = random.Random(seed)
    rng.shuffle(sorted_sessions)

    total_pairs = len(pairs)
    target_val_count = int(val_ratio * total_pairs)

    val_pairs: List[Tuple[Path, Path]] = []
    train_pairs: List[Tuple[Path, Path]] = []

    # Greedily allocate sessions to validation until target is met
    val_count = 0
    val_sessions = set()
    for s in sorted_sessions:
        s_count = len(sessions[s])
        if val_count + s_count <= target_val_count * 1.15 or val_count == 0:
            val_sessions.add(s)
            val_count += s_count
        else:
            break

    for s, s_pairs in sessions.items():
        if s in val_sessions:
            val_pairs.extend(s_pairs)
        else:
            train_pairs.extend(s_pairs)

    return train_pairs, val_pairs


class IRTPavementDataset(Dataset):
    """Paired RGB and Thermal Dataset for Pix2Pix cGAN."""

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        img_size: int = 256,
        is_train: bool = True,
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_path, thm_path = self.pairs[idx]

        # Load RGB
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            raise ValueError(f"Failed to load image: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # Load Thermal
        thm_img = cv2.imread(str(thm_path), cv2.IMREAD_GRAYSCALE)
        if thm_img is None:
            thm_img = cv2.imread(str(thm_path))
            thm_img = cv2.cvtColor(thm_img, cv2.COLOR_BGR2GRAY)

        # Resize to fixed resolution
        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        thm_img = cv2.resize(thm_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            # Synchronous geometric augmentations
            if random.random() > 0.5:
                rgb_img = np.fliplr(rgb_img)
                thm_img = np.fliplr(thm_img)
            if random.random() > 0.5:
                rgb_img = np.flipud(rgb_img)
                thm_img = np.flipud(thm_img)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])
                rgb_img = np.rot90(rgb_img, k)
                thm_img = np.rot90(thm_img, k)

            # Asymmetric optical noise on RGB
            if random.random() > 0.5:
                rgb_img = AsymmetricOpticalNoise.apply_synthetic_shadow(rgb_img)
            if random.random() > 0.6:
                rgb_img = AsymmetricOpticalNoise.apply_solar_glare(rgb_img)

        # Normalization to [-1, 1]
        rgb_tensor = torch.from_numpy(rgb_img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        thm_tensor = torch.from_numpy(thm_img.copy()).unsqueeze(0).float() / 127.5 - 1.0

        return rgb_tensor, thm_tensor
