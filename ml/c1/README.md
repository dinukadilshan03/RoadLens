# Component 1 (C1): Road Damage Detection & Temporal Consolidation

This component implements the full C1 research and deployment lifecycle for the RoadLens platform:
1. **Base Model Training**: Training on international road damage benchmarks (RDD2022).
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
│   ├── hyp_base.yaml              # Hyperparameter configuration for road distress detection
│   └── srilanka_finetune.yaml     # (Planned) Domain adaptation / fine-tuning config
│
├── scripts/                       # Standard execution scripts
│   ├── data_cleaning.py           # Cleans RDD annotations & filters to 4 core classes
│   ├── base_model_training.py     # Base model training script (YOLOv8m/YOLO11m)
│   ├── domain_gap_evaluation.py   # Dual-domain evaluation & domain gap calculation
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

## Standard Workflow Execution

### 1. Data Cleaning
Filter out conversion-specific Class 4, sanitize bounding boxes, and log provenance:
```bash
python ml/c1/scripts/data_cleaning.py \
    --source-dir ml/c1/data/RDD_SPLIT \
    --target-dir ml/c1/data/rdd_cleaned \
    --mode symlink
```

### 2. Base Model Training
Train the baseline YOLOv8m detector on 4-class RDD2022:
```bash
python ml/c1/scripts/base_model_training.py \
    --data ml/c1/configs/rdd2022_4class.yaml \
    --hyp ml/c1/configs/hyp_base.yaml \
    --model yolov8m.pt \
    --epochs 50 \
    --imgsz 800 \
    --batch 16
```

### 3. Domain Gap Evaluation
Measure the cross-domain degradation ($\Delta mAP$) on Sri Lankan test data:
```bash
python ml/c1/scripts/domain_gap_evaluation.py \
    --weights ml/c1/weights/base_model_rdd2022.pt \
    --rdd-data ml/c1/configs/rdd2022_4class.yaml \
    --srilanka-data ml/c1/configs/srilanka_test.yaml \
    --imgsz 800
```
