# C1 Implementation Plan
## Smartphone-Based Road-Damage Detection for Sri Lankan Roads

## 1. Document Purpose

This document defines the implementation activities, outputs, dependencies, evaluation procedures, and completion criteria for Component 1 (C1). C1 will develop and evaluate a road-damage detection system that uses smartphone video captured on Sri Lankan roads.

The implementation will establish a public-data baseline before introducing any adaptation method. The selected model and adaptation strategy will be determined by the literature review, baseline results, and project constraints.

## 2. Scope

### 2.1 Included

- Collection and preparation of smartphone road video.
- Extraction and annotation of representative video frames.
- Detection of a defined set of road-damage classes.
- Evaluation of transfer from public or international data to Sri Lankan imagery.
- Evaluation of one justified adaptation or generalisation method.
- Robustness and practical runtime evaluation.
- A prototype that exports damage detections for use by downstream components.

### 2.2 Excluded

- Pixel-level damage segmentation.
- Depth, area, volume, or severity measurement.
- Roughness or IRI estimation.
- IMU-based ride-quality analysis.
- Maintenance prioritisation or repair recommendations.
- Production integration with road-authority systems.

## 3. System Definition

### 3.1 Input

Forward- or downward-facing smartphone video captured from a vehicle. The capture process shall record, where available, device model, resolution, frame rate, mounting position, route, approximate vehicle speed, date, time, weather, and road type.

### 3.2 Output

Each detection shall contain:

- damage class;
- bounding-box coordinates;
- confidence score;
- frame number or timestamp; and
- optional GPS coordinates or other location metadata.

### 3.3 Processing Flow

```text
Smartphone video
       |
Capture-quality checks and metadata logging
       |
Frame extraction and representative-frame selection
       |
Image preprocessing
       |
Road-damage object detector
       |
Confidence filtering and duplicate handling
       |
Timestamped damage detections
```

## 4. Implementation Strategy

The work shall follow this sequence:

```text
Literature review
       |
Research gap and hypothesis
       |
Dataset and capture protocol
       |
Public-data baseline
       |
Domain-gap measurement
       |
Adaptation method
       |
Controlled experiments
       |
Robustness and runtime evaluation
       |
Prototype and final report
```

The adaptation method shall not be selected solely to maximise a final accuracy value. It must address a limitation identified through the literature review or baseline error analysis.

## 5. Work Packages

### WP1: Literature Review and Research Definition

**Activities**

1. Review road-damage object detection and smartphone-based inspection research.
2. Review cross-domain detection, domain adaptation, and semi-supervised learning methods.
3. Review public datasets, including RDD2022 and other compatible datasets.
4. Record dataset characteristics, damage taxonomies, model types, evaluation methods, and reported limitations in a literature matrix.
5. Define the research question and hypothesis.
6. Establish criteria for selecting the detector and adaptation method.

**Outputs**

- Literature matrix.
- Defined research gap or evaluation problem.
- Candidate damage taxonomy.
- Research question and hypothesis.
- Model-selection criteria.

**Dependency:** None.

**Acceptance criteria:** The selected implementation direction is supported by the reviewed literature and is feasible within the available data, hardware, and time constraints.

### WP2: Dataset and Capture Protocol

**Activities**

1. Define a limited set of visually distinguishable damage classes.
2. Prepare annotation rules for class assignment, bounding boxes, ambiguous cases, and overlapping damage.
3. Define the smartphone mounting and recording procedure.
4. Conduct a pilot collection across multiple routes.
5. Extract representative frames while limiting near-duplicate frames.
6. Annotate the local dataset and review annotation quality.
7. Document privacy, permissions, and anonymisation procedures.

The dataset shall seek variation in road type, pavement type, urban or rural setting, weather, lighting, camera position, vehicle speed, damage type, and damage size.

**Outputs**

- Capture protocol.
- Metadata schema.
- Annotation guide.
- Annotated local dataset.
- Data-coverage and quality report.

**Dependency:** WP1 for the initial taxonomy and data requirements.

**Acceptance criteria:** Classes are consistently annotatable, the pilot covers the required conditions, and the dataset is documented sufficiently to reproduce the collection and annotation process.

### WP3: Public-Data Baseline

**Activities**

1. Select an established multi-class object detector.
2. Prepare the selected public dataset using a documented class mapping.
3. Train or configure the source-domain detector.
4. Record preprocessing, hyperparameters, random seeds, hardware, and model versions.
5. Evaluate the detector on source validation data and held-out Sri Lankan data.
6. Review representative false positives and false negatives.

**Baseline 0**

```text
Public or international labelled data
                 |
                 v
        Source-only detector
                 |
                 v
      Held-out Sri Lankan test set
```

**Outputs**

- Baseline model and configuration.
- Source-domain evaluation results.
- Sri Lankan direct-transfer results.
- Baseline error analysis.
- Domain-gap report.

**Dependencies:** WP1 and WP2.

**Acceptance criteria:** The baseline is reproducible, the local test set is not used for training or model selection, and the transfer performance is reported using the agreed evaluation protocol.

### WP4: Adaptation and Generalisation Method

**Activities**

1. Use the domain-gap report to identify the principal failure modes.
2. Select one primary method justified by the literature and local observations.
3. Implement the selected method using version-controlled code and documented configuration.
4. Where appropriate, prepare unlabelled Sri Lankan frames for domain adaptation or pseudo-labelling.
5. Conduct component-level ablations.

Candidate methods include supervised fine-tuning, condition-aware augmentation, source-data selection or reweighting, feature-level domain alignment, and semi-supervised learning. The final method shall be selected after WP3.

**Outputs**

- Adapted model implementation.
- Training configuration.
- Experiment registry.
- Ablation results.

**Dependency:** WP3.

**Acceptance criteria:** The method has a stated hypothesis, uses the same evaluation split as the baseline, and can be compared without changing unrelated experimental conditions.

### WP5: Controlled Experiments

The following experiment structure shall be used as the initial evaluation design. Label percentages may be adjusted after the pilot, but any changes shall be documented.

| Experiment | Training data | Local labels | Purpose |
|---|---|---:|---|
| E1 | Public or international data only | 0% | Direct-transfer baseline |
| E2 | Public data plus conventional local fine-tuning | 100% of training subset | Supervised adaptation reference |
| E3 | Public data plus selected adaptation method | 0% | Unsupervised adaptation |
| E4 | Adaptation method plus local labels | 10% | Limited-label performance |
| E5 | Proposed method | 5% | Resource-efficient adaptation |
| E6 | Proposed method | 10% | Small-label robustness |
| E7 | Proposed method | 20% | Label-scaling behaviour |

All experiments shall use the same held-out local test set, class definitions, evaluation code, and capture assumptions.

**Outputs**

- Experiment registry.
- Model-comparison table.
- Label-efficiency analysis.
- Ablation study.

**Acceptance criteria:** Results identify whether the selected method improves local performance relative to the direct-transfer and supervised baselines. Negative results shall be retained and documented.

### WP6: Robustness and Failure Analysis

Evaluate the held-out local data by condition where sample counts permit. Analysis categories shall include:

- sunny, overcast, wet, shadowed, and low-light conditions;
- urban, rural, residential, and arterial roads;
- pavement and repair types;
- damage class and approximate damage size;
- device, camera position, and resolution; and
- blur, occlusion, glare, and other capture artefacts.

The analysis shall distinguish annotation ambiguity, capture limitations, domain shift, and model errors. Repeated detections of the same physical defect across adjacent frames shall also be assessed.

**Outputs**

- Condition-wise evaluation tables.
- Error taxonomy.
- Representative error examples.
- Limitations and data-coverage statement.

**Dependency:** WP5.

### WP7: Practical Evaluation and Prototype

**Activities**

1. Evaluate the selected model in the intended processing arrangement: offline computer, local server, or on-device inference.
2. Measure inference latency, FPS, processing time per kilometre, model size, memory use, and hardware requirements.
3. Compare practical performance with the reference configuration.
4. Implement a demonstrable upload-or-process workflow.
5. Display detections over video frames and export structured detection records.

```text
Smartphone video
       |
Frame extraction and quality checks
       |
C1 detection model
       |
Class + bounding box + confidence
       |
Timestamped and optionally geotagged report
```

The prototype may use offline or server-assisted processing. The system shall be presented as decision support and assisted review, not as a replacement for engineering inspection or maintenance judgement.

**Outputs**

- Runtime benchmark report.
- Recommended capture and processing configuration.
- Demonstrable prototype.
- Exportable detection report.

**Dependency:** WP5 and WP6.

**Acceptance criteria:** The prototype processes representative smartphone footage, displays detections, and exports records containing the required output fields.

## 6. Evaluation Protocol

### 6.1 Dataset Splits

Training, validation, and test data shall be separated by road segment, route, or collection session. Random splitting of neighbouring video frames shall not be used because it may place near-identical instances in both training and testing.

The local test set shall remain untouched until final evaluation. Validation data may be used for model selection; test data shall not be used for that purpose.

### 6.2 Detection Metrics

- mAP@50 and mAP@50:95, with IoU definitions stated.
- Per-class precision, recall, F1, and AP.
- Confusion matrix.
- Confidence-threshold policy.
- Test-set size and class distribution.

### 6.3 Practical Metrics

- Inference latency and FPS.
- End-to-end processing time per kilometre.
- Model size and memory use.
- Hardware and input resolution.
- Storage and upload requirements.
- Accuracy retained relative to the reference configuration.

All reported results shall include the model version, dataset split, preprocessing, training budget, hardware, random seed, and local label fraction. Repeated-run variation or uncertainty shall be reported where feasible.

## 7. Risks and Mitigation Measures

| Risk | Mitigation |
|---|---|
| Local data is insufficient or unrepresentative | Begin with a pilot, track coverage, and prioritise condition diversity. |
| Annotation inconsistency | Use an annotation guide, double-annotate a subset, and review disagreements. |
| Severe class imbalance | Conduct targeted collection and report per-class results. |
| Video-frame leakage | Split by route or collection session and retain unseen-road footage. |
| Smartphone variation reduces performance | Standardise the initial protocol and test device variation separately. |
| Accuracy is unsuitable for direct field use | Use assisted review and state operational limitations explicitly. |
| Scope expands beyond available resources | Treat a robust baseline and one justified adaptation method as the minimum contribution. |

## 8. Milestones

| Milestone | Completion condition |
|---|---|
| M1: Research definition | Literature matrix, gap, hypothesis, and taxonomy completed. |
| M2: Dataset readiness | Capture protocol, annotation guide, pilot dataset, and quality report completed. |
| M3: Baseline completed | Reproducible source-only model and local transfer results available. |
| M4: Adaptation selected | Method justified by literature and baseline failure analysis. |
| M5: Experiments completed | Comparison, label-efficiency study, and ablations completed. |
| M6: Evaluation completed | Robustness, failure analysis, and runtime benchmarks completed. |
| M7: Prototype completed | Detection workflow and exportable records demonstrated. |

## 9. Completion Criteria

C1 shall be considered complete when the following conditions are met:

1. The local dataset, taxonomy, annotations, and route-based splits are documented.
2. A reproducible public-data baseline has been evaluated on source and Sri Lankan data.
3. The direct-transfer performance and domain gap have been measured.
4. One literature-justified adaptation or generalisation method has been evaluated against controlled baselines.
5. Robustness, failure modes, and practical runtime have been reported.
6. The prototype produces class, bounding box, confidence, timestamp, and optional location records.
7. Conclusions are limited to the tested conditions and do not claim universal performance or production readiness.
