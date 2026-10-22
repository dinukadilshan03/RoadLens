"""Generates realistic benchmark sample crops for Component C2 demonstration:
1. sample_shadow_crack.jpg   (Longitudinal crack under dynamic tree canopy shadows)
2. sample_glare_alligator.jpg(Alligator crack network under tropical asphalt glare)
3. sample_drainage_grate.jpg (Metallic drainage grate to demonstrate OOD false alarm rejection)
4. sample_pothole_water.jpg  (Water-filled asphalt depression cavity)
"""

from pathlib import Path
import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[3]
ML_SAMPLES_DIR = REPO_DIR / "ml" / "c2" / "data" / "samples"
WEB_SAMPLES_DIR = REPO_DIR / "frontend" / "web" / "public" / "samples" / "c2"

for d in [ML_SAMPLES_DIR, WEB_SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def create_asphalt_background(size=(256, 256), base_val=130):
    noise = np.random.normal(0, 15, size).astype(np.float32)
    bg = (np.full(size, base_val, dtype=np.float32) + noise).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


# 1. Longitudinal Crack under Tree Shadows
def make_shadow_crack():
    img = create_asphalt_background(base_val=135)
    # Draw longitudinal crack line
    pts = np.array([[120, 10], [125, 60], [118, 120], [130, 180], [128, 245]], np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(35, 35, 35), thickness=2)
    # Add hairline branch
    cv2.line(img, (125, 60), (160, 95), (45, 45, 45), 1)

    # Add tree shadow canopy (dark polygon)
    shadow = np.ones((256, 256), dtype=np.float32)
    shadow_pts = np.array([[0, 0], [180, 0], [220, 140], [140, 256], [0, 256]], np.int32)
    cv2.fillPoly(shadow, [shadow_pts], 0.42)
    shadow = cv2.GaussianBlur(shadow, (31, 31), 0)
    shadowed = (img.astype(np.float32) * shadow[:, :, None]).clip(0, 255).astype(np.uint8)
    return shadowed


# 2. Alligator Crack under Glare
def make_glare_alligator():
    img = create_asphalt_background(base_val=125)
    # Draw interconnecting polygonal grid
    for x in range(30, 230, 35):
        for y in range(30, 230, 35):
            dx = int(np.random.randint(-8, 8))
            dy = int(np.random.randint(-8, 8))
            cv2.rectangle(img, (x + dx, y + dy), (x + 30 + dx, y + 30 + dy), (40, 40, 40), 1)

    # Add specular sun glare flare
    y, x = np.ogrid[:256, :256]
    glare = np.exp(-((x - 170)**2 + (y - 90)**2) / (2.0 * (45.0)**2)) * 170.0
    glared = np.clip(img.astype(np.float32) + glare[:, :, None], 0, 255).astype(np.uint8)
    return glared


# 3. Metallic Drainage Grate (OOD Rejection Benchmark)
def make_drainage_grate():
    img = create_asphalt_background(base_val=120)
    # Square metal frame
    cv2.rectangle(img, (40, 40), (216, 216), (65, 65, 70), 3)
    # Intersecting black iron bars (which trick standard RGB models into false alligator crack alarms)
    for i in range(55, 205, 18):
        cv2.line(img, (i, 42), (i, 214), (25, 25, 25), 2)
        cv2.line(img, (42, i), (214, i), (25, 25, 25), 2)
    return img


# 4. Water-Filled Pothole
def make_pothole_water():
    img = create_asphalt_background(base_val=140)
    # Irregular crater depression
    center = (128, 130)
    cv2.ellipse(img, center, (75, 55), 25, 0, 360, (30, 35, 40), -1)
    # Deep water pool inside cavity
    cv2.ellipse(img, (125, 132), (55, 38), 20, 0, 360, (15, 20, 25), -1)
    # Rough fractured rim edges
    cv2.ellipse(img, center, (80, 60), 25, 0, 360, (70, 70, 70), 2)
    return img


samples = {
    "sample_shadow_crack.jpg": make_shadow_crack(),
    "sample_glare_alligator.jpg": make_glare_alligator(),
    "sample_drainage_grate.jpg": make_drainage_grate(),
    "sample_pothole_water.jpg": make_pothole_water(),
}

for name, mat in samples.items():
    cv2.imwrite(str(ML_SAMPLES_DIR / name), mat)
    cv2.imwrite(str(WEB_SAMPLES_DIR / name), mat)
    print(f"Generated benchmark crop: {name}")

print("All benchmark samples successfully generated!")
