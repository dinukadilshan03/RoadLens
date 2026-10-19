# Component C2: Noise-Resilient Road Damage Segmentation Engine

**Researcher:** Methma C A N (IT23245860) – B.Sc. (Hons) in Information Technology Specializing in Data Science, SLIIT  
**Supervisors:** Ms. Thamali Dassanayake & Ms. Hansi De Silva  
**Research Group:** J26-DS-309  
**Proposal Title:** *Software Emulated Linear Transposed Cross Attention Framework for Noise Resilient Pavement Segmentation*

---

## 1. Overview & Architectural Role

Component C2 acts as the **Noise-Resilient Segmentation Engine** within the RoadLens platform:
- **Upstream Handoff (from C1):** Ingests cropped $256 \times 256$ RGB optical distress patches with spatiotemporal metadata (`damage_class`, `external_damage_id`, bounding boxes).
- **Stage 1 (Software Thermal Sensor):** Uses a Conditional GAN (Pix2Pix cGAN) trained on pavement moisture thermodynamics (`IRT_Pavement` dataset) to synthesize $256 \times 256$ pseudo-thermal moisture maps, eliminating expensive physical thermal cameras ($20,000+).
- **Stage 2 (Core Segmenter):** Employs dual MobileNetV3 backbones (RGB + Pseudo-Thermal), a **Linear Transposed Cross-Attention (LTCA)** block reducing attention VRAM from 17 GB to 67 MB, a scale-aware dynamic **Alpha Gating Decoder**, and a **Skeletal Topological Loss** ($\mathcal{L}_{Topology}$).
- **Downstream Handoff (to C3 & C4):** Produces the **4-Part Precision Output Suite**:
  1. **Output A:** 2D High-Precision Binary Mask (for pothole area/volume in C3).
  2. **Output B:** 1-Pixel Continuous Skeleton Centerline (for crack length tracing in C3).
  3. **Output C:** Soft Confidence Map (`.npz` array for uncertainty scoring).
  4. **Output D:** High-Contrast Red Visual Overlay (for municipal engineer inspection in C4 / Web Dashboard).

---

## 2. Directory Layout

```
ml/c2/
├── configs/
│   └── cgan_irt.yaml                 # Stage 1 cGAN hyperparameters & training configuration
├── data/
│   ├── __init__.py
│   └── irt_dataset.py                # IRT_Pavement loader with tropical shadow/glare noise simulation
├── models/
│   ├── __init__.py
│   └── cgan.py                       # UNetGenerator256 & 70x70 PatchGANDiscriminator
├── notebooks/
│   └── 01_train_cgan_irt_colab.ipynb # Google Colab training notebook for Stage 1 cGAN
├── scripts/                          # Training, evaluation, and benchmark CLI scripts
├── weights/                          # Frozen checkpoints & metadata contracts (git-ignored)
└── README.md
```

---

## 3. Training Stage 1 cGAN in Google Colab

The primary workflow for Stage 1 uses the interactive Python notebook:
[`ml/c2/notebooks/01_train_cgan_irt_colab.ipynb`](notebooks/01_train_cgan_irt_colab.ipynb)

### Google Drive Setup:
1. Store your dataset in Google Drive under:
   ```
   My Drive/
   └── C2_Pavement_Research/
       └── datasets/
           └── IRT_Pavement/
               ├── RGB/        # 448 visual road surface images
               └── thermal/    # 448 co-registered infrared thermal images
   ```
2. Open [`01_train_cgan_irt_colab.ipynb`](notebooks/01_train_cgan_irt_colab.ipynb) in **Google Colab**.
3. Select **GPU runtime** (`Runtime > Change runtime type > T4 GPU`).
4. Execute cells sequentially. The notebook will:
   - Mount Google Drive and discover paired images.
   - Partition into 80% train / 20% validation via Route-Based Geographical Partitioning.
   - Train Pix2Pix generator and discriminator with asymmetric optical noise injection.
   - Compute validation metrics (MAE, PSNR, SSIM) and save visual comparison sheets every 5 epochs.
   - Export the best checkpoint (`cgan_generator_best.pth`) and the frozen sensor model (`cgan_frozen_sensor.pth`) along with `cgan_sensor_metadata.json` for Stage 2.
