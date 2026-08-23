# Component 1 Planning Report: Smartphone-Based Road-Damage Detection for Sri Lankan Roads

## 1. Purpose and status of this report

This is an internal research-planning report for Component 1 (C1) of a proposed final-year project. It narrows the project to automated **road-damage detection** from smartphone video. It is not a thesis chapter and does not make final claims of originality, accuracy, or deployment readiness.

Two kinds of statements are used deliberately throughout:

- **Project direction / established for planning:** the intended system, its users, constraints, and experiments proposed by the team.
- **To be validated through literature review and experiments:** claims about research gaps, novelty, relative affordability, performance, and suitability for Sri Lankan conditions.

The broader framework may later consume C1 outputs, but Components 2–4 are outside the scope of this report.

## 2. Context and motivation

Routine road inspection is important for prioritising maintenance and identifying hazards. However, small municipalities and resource-constrained road authorities may not have access to specialised inspection vehicles, high-end cameras, or LiDAR-based systems. A smartphone mounted in or on an inspection vehicle could offer a practical way to collect visual road data because it combines a camera, basic location/sensor capability, storage, and widespread availability.

The research motivation is therefore not simply to demonstrate that a smartphone can record road images. Smartphone-based pavement inspection has already been explored in the wider research field and must be reviewed carefully. The practical question is:

> Can a road-damage detector trained on available data be adapted and evaluated for smartphone footage captured under Sri Lankan road conditions, while retaining a deployment path that smaller road authorities could realistically use?

The intended value lies in combining a low-cost acquisition method with reliable cross-domain detection and a practical inspection workflow.

## 3. Problem statement

Road-damage detection models trained on publicly available datasets may perform poorly when applied directly to local smartphone footage. Differences in pavement materials, patching practices, road markings, tropical weathering, lighting, shadows, traffic, camera mounting, motion blur, and damage appearance can create a domain shift between training imagery and Sri Lankan roads.

At present, it is not yet established whether existing public models transfer adequately to the chosen local context, which damage classes are sufficiently distinguishable, or which adaptation strategy gives the best accuracy-to-cost trade-off. These are research questions to be tested, not assumptions.

## 4. Aim

To develop and evaluate an AI-based road-damage detection approach that uses smartphone video and is adapted to Sri Lankan road environments, producing damage classes and bounding-box locations suitable for a low-cost road-inspection workflow.

## 5. Objectives

1. Define an operational set of road-damage classes that can be identified consistently in smartphone imagery and are meaningful for the intended road-authority use case.
2. Build or curate a representative Sri Lankan smartphone-video dataset, with documented collection conditions and annotated damage instances.
3. Establish a public-data-trained baseline and measure how it transfers to held-out Sri Lankan road footage.
4. Investigate and compare feasible adaptation/generalisation strategies using local labelled data and, where appropriate, unlabelled local data.
5. Evaluate detection quality, robustness, inference practicality, and failure modes under realistic local conditions.
6. Specify how C1 detections can be exported to the wider framework as timestamped, optionally geotagged, damage records with class, confidence, and bounding box.

## 6. C1 system definition

### Input

A forward- or downward-facing smartphone video stream captured from a vehicle. The capture protocol should record relevant metadata where possible: device model, camera resolution, frame rate, mount position, approximate vehicle speed, route, time of day, weather, and road type.

### Output

For each selected frame or video segment, C1 should output one or more records containing:

- damage type/class;
- bounding-box coordinates in the image;
- detection confidence;
- video frame or timestamp;
- optional GPS/location information when available; and
- optional track/instance identifier if temporal tracking is later included.

### Interface to the wider framework

C1 provides localised damage detections. Downstream components may use these detections for segmentation, severity measurement, or road-condition reporting, but C1 will not perform those functions in this study.

## 7. Proposed pipeline

```text
Smartphone video
      |
Capture-quality checks and metadata logging
      |
Frame sampling / extraction
      |
Image preprocessing or augmentation strategy
      |
Road-damage object detector
      |
Post-processing: confidence filtering, duplicate handling,
optional temporal consistency and timestamp/GPS association
      |
Damage detections: class + bounding box + confidence + time/location
```

The early implementation should remain simple enough to establish a defensible baseline. More advanced video-level processing—such as temporal tracking or multi-frame fusion—should be treated as optional enhancement work only if it addresses a measured failure mode.

## 8. Research gap and novelty framing

The project should avoid claiming that using a smartphone alone is novel. A more defensible potential contribution is:

> A systematically evaluated adaptation and deployment-oriented road-damage detection workflow for smartphone footage in Sri Lankan road environments.

Potential research questions include:

- How large is the performance gap when a detector trained on public/international data is applied to Sri Lankan smartphone video without local adaptation?
- How much local labelled data is needed to obtain meaningful improvement?
- Which feasible strategy—fine-tuning, domain-aware augmentation, semi-supervised learning, domain adaptation, or another approach identified through literature review—best improves local performance under the project constraints?
- Which capture conditions and road characteristics most affect reliability?

Whether these questions represent an underexplored gap, and whether the selected approach is novel at a national or international level, must be determined through a structured literature review. The final contribution may be methodological, empirical, dataset-oriented, deployment-oriented, or a combination; it should be stated only after that review and experimental results.

## 9. Likely local operating challenges to investigate

The following are hypotheses and data-collection factors, not yet verified findings:

- varied pavement textures, resurfacing, repairs, and patching that resemble damage;
- cracks, potholes, edge breaks, rut-like wear, and surface deterioration with inconsistent visual appearance;
- strong sun, tree/building shadows, rain, wet pavement, glare, and low-light conditions;
- occlusion by vehicles, pedestrians, parked vehicles, debris, and water;
- motion blur, vibration, camera-angle variation, dirty lenses, and compression artifacts from smartphones;
- road markings, stains, utility covers, leaves, and repairs that can cause false positives;
- imbalance between common minor defects and rarer but important severe defects; and
- variation across urban, rural, residential, and arterial roads.

The study should record these factors so that results can be broken down by condition rather than reported only as a single aggregate score.

## 10. Data strategy

### Public/source data

Public road-damage datasets can support initial development and pre-training. Each candidate dataset should be screened for image source, label definitions, licence, road geography, annotation quality, and compatibility with the local class taxonomy. Public labels should not be assumed to map directly to local maintenance terminology.

### Local Sri Lankan data

Collect smartphone videos using a repeatable mounting and capture protocol. The initial dataset should seek diversity rather than only volume: multiple locations, pavement types, road classes, lighting/weather conditions, device configurations where feasible, and damage severities.

Annotations should use explicit class definitions, bounding-box rules, treatment of ambiguous cases, and a quality-control procedure. A small double-annotated subset and agreement review would help reveal unclear labels early.

### Dataset splits and leakage prevention

Training, validation, and test splits should be separated by road segment, route, or collection session—not random neighbouring frames. Random frame splitting can place visually near-identical damage instances in both training and testing and overstate performance. A held-out test set should represent unseen roads and, where possible, challenging conditions.

Privacy and permissions for video collection should be addressed before collection. Where imagery includes faces, licence plates, or identifiable private property, the project should use an appropriate handling policy consistent with institutional ethics requirements.

## 11. Model and training strategy

The architecture should not be locked before the literature review and a baseline feasibility study. C1 needs a detector capable of multi-class object detection and bounding-box output, with a realistic balance between accuracy, inference speed, model size, and available hardware.

The conceptual experimental progression is:

1. **Direct-transfer baseline:** train or start from a model using public/international data, then evaluate without local adaptation on the Sri Lankan test set.
2. **Local fine-tuning baseline:** fine-tune using an increasing amount of labelled Sri Lankan data.
3. **Proposed adaptation/generalisation methods:** test one or a small number of approaches selected from literature and justified by local observations. Candidates may include condition-aware data augmentation, self-/semi-supervised use of unlabelled local video, feature-level domain adaptation, or source-data selection/reweighting.
4. **Practical variant:** assess a compressed, lightweight, or lower-resolution configuration if the selected detector is otherwise unsuitable for the intended workflow.

All variants should use the same held-out local test set, class definitions, and capture assumptions. The main research result should be a controlled comparison, not a claim based solely on a single model's accuracy.

## 12. Evaluation methodology

### Primary detection measures

- mean Average Precision (mAP) at clearly stated IoU thresholds;
- per-class precision, recall, and F1 score;
- confusion matrix and qualitative false-positive/false-negative review; and
- class frequency and annotation coverage.

### Robustness and practical measures

- performance by road type, lighting/weather category, device/camera setup, and damage size where sample counts permit;
- inference time, frames per second, model size, memory use, and hardware used;
- end-to-end video processing throughput; and
- temporal duplicate behaviour: repeated detections of the same physical defect across adjacent frames.

### Baselines

At minimum, compare:

| Experiment | Purpose |
|---|---|
| Public-data/direct-transfer model on local test data | Quantifies initial domain-transfer gap. |
| Same detector fine-tuned on local labelled data | Measures the value of local supervision. |
| Proposed adaptation/generalisation method | Tests the central research hypothesis. |
| Optional lightweight/deployment configuration | Shows accuracy–practicality trade-off. |

Reported results should include uncertainty or repeated-run variation where feasible, plus test-set size and class distribution. A practical claim should be tied to actual measured runtime and hardware, not inferred from model type alone.

## 13. Practical deployment considerations

The target workflow is deliberately modest: mount a smartphone, drive an assigned route, upload or process video, and receive reviewed damage detections. A first version need not require real-time inference on the phone. Offline or server-assisted processing may be more achievable while retaining low-cost data acquisition.

Deployment design questions to decide through prototyping include:

- whether processing occurs on-device, on a local computer, or through a server;
- a recommended phone mount, orientation, height, and capture resolution/frame rate;
- acceptable driving-speed and lighting conditions;
- how staff review uncertain detections before maintenance decisions;
- storage/upload requirements and intermittent-connectivity handling; and
- a simple export format for reports or mapping tools.

The system should be positioned as decision support and data collection assistance. It should not be represented as replacing safety inspection, engineering judgement, or a road authority's maintenance-prioritisation process.

## 14. Scope boundaries

C1 includes visual detection and classification of predefined road-damage categories from smartphone video.

C1 excludes:

- pixel-level damage segmentation;
- geometric depth, area, volume, or severity measurement;
- roughness/IRI estimation or IMU-based ride-quality assessment;
- final maintenance prioritisation, cost estimation, or repair recommendation;
- a claim of universal performance across all Sri Lankan roads; and
- production deployment or integration with authority systems beyond a demonstrable output workflow.

The damage taxonomy should remain limited enough for consistent annotation and statistically meaningful evaluation. Adding many visually similar or rare classes prematurely is a major feasibility risk.

## 15. Key risks and mitigations

| Risk | Planned mitigation |
|---|---|
| Insufficient or unrepresentative local data | Start collection early; prioritise road/condition diversity; document coverage gaps. |
| Ambiguous or inconsistent labels | Define annotation guide; train annotators; review disagreements and edge cases. |
| Severe class imbalance | Use targeted collection, appropriate sampling/loss strategies, and per-class reporting. |
| Overfitting or data leakage from video frames | Split by route/session; retain unseen-road test footage. |
| Model accuracy is unsuitable for direct field use | Reframe output as assisted review; analyse failure modes; refine scope and capture protocol. |
| Smartphone-video variability overwhelms the model | Standardise mount/capture settings for the first study; test controlled variations later. |
| Scope becomes too broad | Treat a robust local baseline and one justified adaptation method as the minimum viable research contribution. |

## 16. Expected contribution

If validated, C1 is expected to contribute:

1. a clearly specified smartphone-video road-damage detection workflow for the selected Sri Lankan context;
2. a curated and documented local evaluation dataset, subject to permissions and dissemination constraints;
3. empirical evidence on cross-domain transfer from public road-damage data to local imagery;
4. a controlled evaluation of locally appropriate adaptation/generalisation strategy or strategies; and
5. a practical detection-output interface for the broader road-condition framework.

The contribution should ultimately be expressed in evidence-based terms. For example, the project may be able to claim measured improvement over a direct-transfer baseline under defined conditions; it should not claim broad novelty, affordability, or authority readiness without supporting literature, cost analysis, and field evaluation.

## 17. Immediate next steps

1. Conduct a focused literature review on road-damage object detection, smartphone-based acquisition, cross-domain evaluation, and relevant local road-inspection practice.
2. Select a small, operational damage taxonomy and write the annotation guide.
3. Design and pilot the smartphone capture protocol on a small number of routes.
4. Identify public datasets and create the direct-transfer baseline.
5. Annotate a local pilot set, inspect class balance and annotation consistency, then refine data collection.
6. Finalise research questions and select one adaptation strategy only after baseline transfer results are available.
