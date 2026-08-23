

## 1. First, don't start with the "novel model"

Start by building a **strong baseline**.

Your C1 pipeline would initially be:

```text
Smartphone video
      ↓
Frame extraction
      ↓
Road-damage detector
      ↓
Damage class + bounding box
      ↓
Evaluation on Sri Lankan data
```

Use an established object detector as the baseline. The exact YOLO version can be decided after your literature review.

The important thing is that the baseline answers:

> **How well does an existing road-damage detector trained on international data perform on Sri Lankan roads without adaptation?**

That becomes your **Baseline 0**.

---

# 2. Build your dataset strategy

This is probably more important than the model initially.

You can have:

### Source domain

Existing datasets such as RDD2022:

```text
RDD2022
├── Japan
├── India
├── China
├── Czech Republic
├── Norway
└── United States
```

These give you relatively large amounts of labelled road-damage data.

### Target domain

Your own Sri Lankan smartphone recordings:

```text
Sri Lankan roads
        ↓
Smartphone video
        ↓
Extract frames
        ↓
Select representative frames
        ↓
Annotate damage
        ↓
Sri Lankan dataset
```

Your target dataset should ideally contain variation in:

- road type
- pavement type
- urban/rural environment
- weather
- lighting
- camera position
- vehicle speed
- damage type
- damage severity

You don't necessarily need thousands of hours of video.

A carefully designed dataset of **representative frames** can be much more useful.

---

# 3. Establish the "domain gap"

Before implementing adaptation, actually measure the problem.

Train your baseline on the source data:

```text
RDD2022
   ↓
Baseline detector
```

Then evaluate:

```text
             Performance
                 │
        ┌────────┴────────┐
        ▼                 ▼
International        Sri Lankan
validation           validation
```

Suppose you get something hypothetical like:

| Dataset | mAP@50 |
|---|---:|
| RDD2022 validation | 86% |
| Sri Lankan test set | 58% |

That difference is extremely useful.

You can now demonstrate:

> **The detector performs well in the source domain but experiences significant performance degradation in the Sri Lankan target domain.**

Now you have empirical evidence for your research problem.

---

# 4. Then implement your first adaptation method

Suppose your literature review identifies:

> Feature-level domain alignment

as an appropriate gap/direction.

Your system becomes:

```text
                     SOURCE
                  RDD2022 images
                       │
                       ▼
                ┌─────────────┐
                │   Detector  │
                └──────┬──────┘
                       │
                  Features
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        Detection loss    Domain loss
              │                 │
              └────────┬────────┘
                       ▼
                 Adapted model
                       │
                       ▼
                Sri Lankan roads
```

The important part is that you aren't inventing everything from scratch.

You take an established technique from the literature and adapt/extend it to your specific road-damage problem.

---

# 5. A potentially stronger direction: exploit unlabeled Sri Lankan video

This is where your smartphone approach becomes particularly interesting.

Imagine you collect:

**20 hours of Sri Lankan road video.**

You don't want to manually annotate every frame.

Instead:

```text
Sri Lankan video
       │
       ▼
Unlabelled target frames
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
 Domain adaptation      Pseudo-labeling
       │                     │
       └──────────┬──────────┘
                  ▼
             Improved model
                  │
                  ▼
       Small labelled Sri Lankan
              test set
```

You could potentially have:

- large labelled international dataset
- large unlabelled Sri Lankan dataset
- relatively small labelled Sri Lankan dataset

That is a very realistic road-authority scenario.

And it gives you a much stronger research question:

> **Can a road-damage detector be adapted to Sri Lankan road conditions using large amounts of inexpensive smartphone video and only limited local annotation?**

That is substantially more interesting than simply fine-tuning YOLO on Sri Lankan images.

---

# 6. Your experiments become the actual research

This is crucial.

You shouldn't have only:

> Proposed model → 91% mAP.

You should have an **ablation and comparison study**.

For example:

| Experiment | Training | Sri Lankan labels | Purpose |
|---|---|---:|---|
| E1 | International only | 0 | Baseline |
| E2 | International + conventional fine-tuning | 100% | Supervised adaptation |
| E3 | Domain adaptation | 0 | Test unsupervised adaptation |
| E4 | Domain adaptation | 10% | Limited-label scenario |
| E5 | Proposed method | 5% | Resource-efficient adaptation |
| E6 | Proposed method | 10% | Robustness |
| E7 | Proposed method | 20% | Scaling behaviour |

Now you can answer something scientifically meaningful:

> How much local annotation is actually required?

That aligns **very well** with your resource-constrained road-authority motivation.

---

# 7. Test generalization, not just accuracy

Suppose your model achieves 90% mAP on your Sri Lankan test set.

That's not enough.

You want to test whether it works under different conditions:

```text
Sri Lankan test data
│
├── Sunny
├── Overcast
├── Wet
├── Shadows
├── Urban
├── Rural
├── Good pavement
├── Rough pavement
├── Small cracks
└── Severe damage
```

You could produce something like:

| Condition | Baseline | Proposed |
|---|---:|---:|
| Sunny | 74 | 87 |
| Cloudy | 69 | 84 |
| Wet | 51 | 76 |
| Shadows | 55 | 79 |
| Small cracks | 43 | 68 |
| Potholes | 72 | 86 |

Those numbers are just examples, but **this type of experiment is what would establish your contribution**.

---

# 8. Then address the smartphone constraint

After you have the research model, test the practical system.

You could evaluate:

### Accuracy

- mAP@50
- mAP@50:95
- precision
- recall
- F1
- per-class AP

### Real-world performance

- FPS
- inference latency
- processing time per kilometre
- memory consumption
- model size

### Robustness

- phone models
- camera resolution
- camera position
- vehicle speed
- lighting
- weather

This allows you to make a practical statement such as:

> The proposed method maintains X% of its desktop performance while operating under smartphone-based data acquisition.

---

# 9. Eventually build the actual prototype

The final C1 prototype doesn't need to be a sophisticated mobile application.

A practical prototype could be:

```text
             Smartphone
                 │
          Record video
                 │
                 ▼
          Upload/process
                 │
                 ▼
        C1 Detection Model
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
    Damage              Bounding
     type                  box
       │                   │
       └─────────┬─────────┘
                 ▼
          Detection report
```

For your proposal/prototype presentation, you could show:

**Input**

Smartphone road video.

**Processing**

Detection model.

**Output**

```text
Frame 1423

┌──────────────────────────────┐
│                              │
│       ROAD                   │
│                              │
│   ┌──────────────┐           │
│   │ ALLIGATOR    │           │
│   │ CRACK        │           │
│   └──────────────┘           │
│                              │
└──────────────────────────────┘
```

with confidence and class information.

---

# 10. The research development cycle

So your actual workflow should be:

```text
Literature review
       ↓
Identify limitation/gap
       ↓
Formulate research question
       ↓
Collect source + Sri Lankan data
       ↓
Build baseline
       ↓
Measure domain gap
       ↓
Implement adaptation method
       ↓
Run controlled experiments
       ↓
Ablation study
       ↓
Robustness testing
       ↓
Smartphone/practical evaluation
       ↓
Prototype
       ↓
Research conclusions
```

### The most important principle

**Don't decide your novelty first and then force an implementation around it.**

Instead:

> **Literature → gap → hypothesis → method → experiment → evidence**

For example:

**Literature says:**  
Existing detectors trained on international road datasets suffer cross-country degradation.

**Your observation:**  
Sri Lankan roads have different visual characteristics and are absent from major multinational datasets.

**Research question:**  
Can domain adaptation reduce this performance degradation using limited/no Sri Lankan labels?

**Hypothesis:**  
Target-domain adaptation will improve Sri Lankan detection performance over a source-only detector.

**Method:**  
Implement and extend an appropriate domain-adaptation technique.

**Experiment:**  
Compare source-only, fine-tuned, existing adaptation, and proposed adaptation.

**Evidence:**  
mAP/precision/recall + per-condition performance + label-efficiency analysis.

**Contribution:**  
A validated adaptation strategy for low-cost smartphone-based road-damage detection under Sri Lankan conditions.

That is the difference between **"building an AI model"** and doing a **research project**.

For C1 specifically, I would spend the next stage **not coding yet**. Read the five core papers, build a literature matrix, and identify exactly what **DA-RDD and the 2026 cross-domain work still don't solve**. Once that is clear, choosing the model and implementation becomes much easier.