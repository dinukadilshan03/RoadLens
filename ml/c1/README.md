# Component 1 (C1): Road Damage Detection & Temporal Consolidation

This component implements the full C1 research and deployment lifecycle for the RoadLens platform:
1. **Base Model Pretraining**: Training on international road damage benchmarks (RDD2022).
2. **Domain Gap Evaluation**: Proving cross-domain degradation when zero-shot evaluated on Sri Lankan roads.
3. **Domain Adaptation & Fine-Tuning**: Adapting the base detector to local Sri Lankan road features.
4. **Platform Inference & Temporal Tracking**: Deploying the detector with multi-frame tracking (ByteTrack/BoT-SORT) for temporal defect consolidation and feeding cropped defect regions to downstream C2 (Crack Segmentation).

---

## Component Architecture & Structure

```text
ml/c1/
├── README.md                      # Lifecycle documentation & CLI commands
├── .gitignore                     # Ignores weights (*.pt, *.onnx), runs/, and raw datasets
│
├── configs/                       # Dataset & experiment configurations
│   ├── rdd2022_4class.yaml        # Base model training config (RDD2022 international)
│   ├── srilanka_test.yaml         # Target domain evaluation config (Sri Lankan roads)
│   └── srilanka_finetune.yaml     # (Planned) Domain adaptation / fine-tuning config
│
├── scripts/                       # Modular CLI execution scripts
│   ├── prepare_rdd_4class.py      # Cleans RDD annotations & filters to 4 core classes
│   ├── train.py                   # Base model training script (YOLOv8m/YOLO11m)
│   ├── evaluate.py                # Dual-domain evaluation & domain gap calculation
│   ├── finetune.py                # (Planned) Local road fine-tuning / domain adaptation
│   └── export.py                  # (Planned) ONNX / TorchScript model export
│
├── adapters/                      # Production RoadLens runtime integration
│   ├── __init__.py
│   ├── detector.py                # (Planned) YOLO inference returning standard RoadLens bboxes
│   └── tracker.py                 # (Planned) Temporal defect identity consolidation (C1 -> C2)
│
├── notebooks/                     # Exploratory analysis & research reporting
│   ├── data_exploration.ipynb     # 3-stage RDD2022 exploratory data analysis
│   └── domain_gap_analysis.ipynb  # (Planned) Cross-domain gap charts & failure analysis
│
├── weights/                       # Model checkpoints (.pt, .onnx) - git-ignored
│   └── .gitkeep
│
└── runs/                          # Training logs, validation curves, and plots - git-ignored
    └── .gitkeep
```

---

## Active Phase: Base Model Training

### 1. Prepare 4-Class Dataset
Filter out conversion-specific Class 4 and sanitize bounding boxes:
```bash
python ml/c1/scripts/prepare_rdd_4class.py \
    --source-dir ml/c1/data/RDD_SPLIT \
    --target-dir ml/c1/data/RDD_SPLIT_4CLASS
```

### 2. Train Base International Model
Train the baseline YOLOv8m detector on RDD2022:
```bash
python ml/c1/scripts/train.py \
    --data ml/c1/configs/rdd2022_4class.yaml \
    --model yolov8m.pt \
    --epochs 50 \
    --imgsz 800 \
    --batch 16
```

### 3. Evaluate Cross-Domain Gap
Measure the performance drop ($\Delta mAP$) on Sri Lankan test data:
```bash
python ml/c1/scripts/evaluate.py \
    --weights ml/c1/runs/train/rdd2022_base_model/weights/best.pt \
    --rdd-data ml/c1/configs/rdd2022_4class.yaml \
    --srilanka-data ml/c1/configs/srilanka_test.yaml \
    --imgsz 800
```

