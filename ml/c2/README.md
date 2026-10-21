# Component C2: Noise-Resilient Road Damage Segmentation Engine

**Researcher:** Methma C A N (IT23245860) – B.Sc. (Hons) in Information Technology Specializing in Data Science, SLIIT  
**Supervisors:** Ms. Thamali Dassanayake & Ms. Hansi De Silva  
**Research Group:** J26-DS-309  
**Proposal Title:** *Software Emulated Linear Transposed Cross Attention Framework for Noise Resilient Pavement Segmentation*

---

## 1. Overview & Architectural Role

Component C2 serves as the **Noise-Resilient Segmentation Engine** within the RoadLens platform:
- **Upstream Ingestion (from C1):** Ingests cropped $256 \times 256$ RGB optical distress patches with spatiotemporal metadata (`damage_class`, `external_damage_id`, bounding boxes).
- **Stage 1 (Software Thermal Sensor):** Uses a Conditional GAN (Pix2Pix cGAN) trained on pavement moisture thermodynamics (`IRT_Pavement` dataset) to synthesize $256 \times 256$ pseudo-thermal moisture maps in pure software, eliminating expensive physical thermal cameras ($20,000+).
- **Stage 2 (Core Segmenter):** Employs dual MobileNetV3 backbones (Visual RGB + Pseudo-Thermal), a **Linear Transposed Cross-Attention (LTCA)** block reducing attention VRAM from 17 GB to 67 MB, a scale-aware dynamic **Alpha Gating Decoder**, and specialized loss functions:
  - **Cracks:** Skeletal Topological Loss ($\mathcal{L}_{Topology}$) with differentiable soft-skeletonization for unbroken 1-pixel centerlines.
  - **Potholes:** Region-Overlap BCE + Soft Dice + Boundary Edge Loss for precise 2D surface area and crater perimeter delineation.
- **Downstream Handoff (Smart Routing Switch to C3 & C4):** Produces the **4-Part Precision Output Suite**:
  1. **Output A:** 2D High-Precision Binary Mask (for pothole area/volume in C3).
  2. **Output B:** 1-Pixel Skeleton Centerline / Rim Contour (for crack length tracing in C3).
  3. **Output C:** Soft Confidence Map (`.npz` array for depth weighting in C3).
  4. **Output D:** High-Contrast Red Visual Overlay (for municipal engineer inspection in C4 / Web Dashboard).

---

## 2. Directory Layout

```
ml/c2/
├── configs/
│   └── cgan_irt.yaml                         # Stage 1 cGAN hyperparameters
├── data/
│   ├── __init__.py
│   ├── irt_dataset.py                        # IRT_Pavement loader with noise simulation
│   └── segmentation_dataset.py               # Core segmentation & pothole loader
├── models/
│   ├── __init__.py
│   ├── alpha_gate.py                         # Dynamic Alpha attention gate module
│   ├── cgan.py                               # UNetGenerator256 & PatchGANDiscriminator
│   ├── ltca.py                               # Linear Transposed Cross-Attention (LTCA)
│   ├── segmenter.py                          # DualStreamLTCASegmenter network
│   └── topological_loss.py                   # Skeletal topological loss & soft-clDice
├── notebooks/
│   ├── 01_train_cgan_irt_colab.ipynb         # Stage 1: Pix2Pix cGAN on IRT_Pavement
│   ├── 02_train_core_segmentation_colab.ipynb# Stage 2: Dual-Stream LTCA on Core_Segmentation
│   └── 03_train_pothole_segmentation_colab.ipynb # Stage 2: Dual-Stream LTCA on Pothole segmentation
├── pipeline/
│   ├── __init__.py
│   └── output_suite.py                       # 4-Part Output Suite generator & routing switch
├── weights/                                  # Trained checkpoints (git-ignored)
└── README.md
```

---

## 3. Google Colab Training Notebooks

### Stage 1: Software Thermal Sensor (cGAN)
- **Notebook:** [`01_train_cgan_irt_colab.ipynb`](notebooks/01_train_cgan_irt_colab.ipynb)
- **Dataset:** `My Drive/C2_Pavement_Research/datasets/IRT_Pavement/` (`RGB` and `thermal`)
- **Output Weights:** `cgan_frozen_sensor.pth`

### Stage 2 (Potholes): Pothole Segmentation Engine
- **Notebook:** [`03_train_pothole_segmentation_colab.ipynb`](notebooks/03_train_pothole_segmentation_colab.ipynb)
- **Dataset:** `My Drive/C2_Pavement_Research/datasets/Pothole segmentation/` (`train`, `val`, `test`)
- **Key Features:**
  - Loads frozen Stage 1 `cgan_frozen_sensor.pth` to generate moisture void maps.
  - Dual-Stream MobileNetV3 + LTCA ($O(N)$ channel matching, 67 MB VRAM).
  - Region-Overlap BCE + Dice + Boundary Edge Loss.
  - Automatically exports 4-Part Output Suite (Output A: 2D Binary Mask, Output B: Rim Contour, Output C: Soft Confidence NPZ, Output D: Red Visual Overlay) for C3 3D volume integration.
