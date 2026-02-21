# 🦴 Chronos-MSK: Explainable Bone Age Assessment at the Edge

**Bias-aware skeletal maturity assessment with calibrated confidence using MedSigLIP and MedGemma — running entirely offline on consumer hardware.**

---

## Overview

Chronos-MSK is a multi-agent AI system for pediatric bone age estimation that achieves **8.81-month MAE** (surpassing typical human expert variability of 10-13 months) while providing:

- **Calibrated confidence** — HIGH-confidence predictions achieve 6.96-month MAE (83.9% within ±12 months)
- **Clinical narratives** — MedGemma generates radiologist-quality reports explaining each prediction
- **Demographic fairness** — Retrieval stratified by sex and race using the Digital Hand Atlas
- **Edge deployment** — Runs fully offline on a consumer GPU (less than 6 GB VRAM)

---

## Results

### Core Metrics

Evaluated on 1,425 held-out RSNA validation cases:

| Metric | Value |
|---|---|
| **MAE** | **8.81 months (0.73 years)** |
| Median Absolute Error | 7.27 months |
| RMSE | 11.39 months |
| Pearson r | 0.963 |
| R² | 0.927 |
| Within ±6 months | 42.9% |
| Within ±12 months | 73.2% |
| Within ±24 months | 95.7% |

### Confidence Calibration

The system provides calibrated confidence tiers validated with a positive monotonic correlation (r = +0.20):

| Tier | Cases | MAE | Within ±12m |
|---|---|---|---|
| HIGH | 535 (37.5%) | 6.96m | 83.9% |
| MODERATE | 644 (45.2%) | 9.69m | 67.2% |
| LOW | 245 (17.2%) | 10.71m | 64.5% |

### Sex-Stratified Performance

| Sex | Cases | MAE |
|---|---|---|
| Male | 773 | 8.26 months |
| Female | 652 | 9.47 months |

### Age-Stratified Performance

| Age Range | Cases | Regressor MAE | Atlas MAE |
|---|---|---|---|
| 0-5 years | 94 | 9.54m | 28.49m |
| 5-10 years | 394 | 10.04m | 31.33m |
| 10-15 years | 809 | 8.07m | 12.68m |
| 15-19 years | 124 | 8.45m | 15.81m |

---

## Architecture

The system orchestrates six specialized agents, each with a distinct validated role:

```
X-Ray Image + Sex
        |
        v
+----------------+
|  Agent 1:      |   YOLOv8
|  Scout         |   Distal radius detection
|  (Detection)   |   Rotation-invariant, 15% padded crop
+-------+--------+
        |
        v
+----------------+          +---------------------+
|  Agent 2:      |          |  Agent 3:           |
|  Radiologist   |          |  Archivist          |
|  (MedSigLIP    |          |  (MedSigLIP +       |
|   + SVM)       |          |   FAISS retrieval)  |
|                |          |                     |
|  TW3 Staging   |          |  Visual Twins +     |
+-------+--------+          |  Confidence Signal  |
        |                   +----------+----------+
        v                              |
+----------------+                     |
|  Agent 5:      |                     |
|  Regressor     |                     |
|  (MedSigLIP    |                     |
|   + LoRA)      +----------+         |
|                |          |         |
|  228 month-bin |          v         v
|  distribution  |   +----------------+
+----------------+   |  Agent 6:      |
                     |  Ensemble      +----> Final Age (months)
                     |  (Calibrated)  |
                     +-------+--------+
                             |
                             v
                     +----------------+
                     |  Agent 4:      |
                     |  Narrator      +----> Clinical Report
                     |  (MedGemma     |
                     |   1.5 4B-IT)   |
                     +----------------+
```

### Agent Details

| Agent | Model | Function |
|---|---|---|
| **Scout** | Custom YOLOv8 | Rotation-invariant distal radius detection with 15% padded crop |
| **Radiologist** | MedSigLIP-448 + SVM | TW3 maturity staging from frozen 1152-D embeddings |
| **Archivist** | MedSigLIP-448 + SOTA Projector + FAISS | Demographic-stratified Visual Twin retrieval |
| **Regressor** | MedSigLIP-448 + LoRA | Softmax distribution over 228 month-bins with sex conditioning |
| **Ensemble** | Rule-based | Distance-calibrated confidence with age-aware gating |
| **Narrator** | MedGemma 1.5 4B-IT | Clinical narrative report generation anchored to regressor |

---

## HAI-DEF Models Used

### MedSigLIP-448 — Three Roles, One Backbone

MedSigLIP-448 serves as the unified visual backbone across three distinct functions:

1. **Feature extraction**: Frozen encoder produces 1152-D embeddings for classification and retrieval
2. **Regression backbone**: LoRA-adapted with a 4-layer head incorporating sex conditioning for age estimation
3. **Atlas embedding engine**: Full-image embeddings projected through a trained 256-D metric space for demographic-partitioned nearest-neighbor search

This triple utilization means the 1.2 GB base model supports all three downstream tasks through lightweight adapters and heads totaling only 24 MB.

### MedGemma 1.5 4B-IT — The Reasoning Layer

MedGemma receives structured evidence (regressor estimate, atlas matches with distances, confidence tier) alongside the X-ray image and generates formal radiology reports with Findings, Impression, and Bone Age Assessment sections. It does not override the quantitative prediction — it explains it.

Key behavior (validated on 1,425 cases):
- Agrees with regressor 76% of the time
- Output clamped to ±12 months as safety bound
- Only 3% of cases required clamping
- Produces radiologist-quality explanations of ossification patterns

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Weights Setup

Download pretrained weights and place them in the `weights/` directory:

```
weights/
├── best_scout.pt              # YOLOv8 distal radius detector
├── radiologist_head.pkl       # SVM classifier for TW3 staging
├── projector_sota.pth         # SOTA demographic projector (1152->256D)
└── medsiglip_sota/
    ├── config.json            # Regressor configuration
    ├── heads.pth              # Regression heads
    └── adapter/               # LoRA adapter weights
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

### Build Retrieval Index

Embed the Digital Hand Atlas images and build FAISS indices:

```bash
python train_retriever_sota.py
```

This will:
1. Scan the atlas directory structure
2. Embed all images with MedSigLIP (cached after first run)
3. Train the demographic projector with SOTA multi-loss approach
4. Rebuild FAISS indices automatically

### Run the Gradio Demo

```bash
python app.py
```

Navigate to `http://localhost:7860` to access the web interface.

### Run Evaluation

```bash
# Fast evaluation — ensemble only, no VLM (takes ~30 minutes)
python eval_embedding_fix.py

# Full evaluation with MedGemma narratives (requires LM Studio running)
python vlm_eval.py

# Compute competition metrics summary
python compute_competition_metrics.py
```

---

## Project Structure

```
chronos-msk/
│
├── app.py                          # Gradio demo application
├── main.py                         # CLI pipeline entry point
│
├── agents/
│   ├── __init__.py
│   ├── agent1_scout.py             # YOLOv8 radius detection
│   ├── agent2_radiologist.py       # MedSigLIP + SVM staging
│   ├── agent3_archivist.py         # FAISS retrieval with SOTA projector
│   ├── agent4_vlm_client.py        # MedGemma VLM client with clamping
│   ├── agent5_regressor.py         # MedSigLIP + LoRA regression
│   └── agent6_ensemble.py          # Confidence-calibrated ensemble
│
├── train_retriever_sota.py         # SOTA retrieval training pipeline
├── eval_embedding_fix.py           # Ensemble evaluation script
├── eval_ensemble.py                # Quick ensemble evaluation
├── vlm_eval.py                     # Full pipeline + VLM evaluation
├── compute_competition_metrics.py  # Metrics summary for writeup
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Datasets

### RSNA Pediatric Bone Age (14,236 images)

Primary training and validation source. We used a held-out 1,425-case validation set with strict no-leakage protocol. Labels are bone age in months with sex metadata.

### USC Digital Hand Atlas (1,390 images)

Explicitly designed for ethnic diversity with even distribution across four racial groups:

| Group | Male | Female | Total |
|---|---|---|---|
| Asian | 167 | 167 | 334 |
| Black | 184 | 174 | 358 |
| Caucasian | 167 | 166 | 333 |
| Hispanic | 182 | 183 | 365 |

FAISS indices are partitioned by Sex and Race (8 partitions), ensuring Visual Twins come from biologically relevant populations.

---

## Training Details

### Regressor (Agent 5)

- **Base**: MedSigLIP-448 vision encoder
- **Adaptation**: LoRA (Low-Rank Adaptation)
- **Head**: 4-layer MLP with sex conditioning via learned embedding
- **Output**: Softmax over 228 bins, expected value gives age in months
- **TTA**: Original + horizontal flip, averaged

### Retrieval Projector (Agent 3)

The SOTA demographic projector (1152-D to 256-D) is trained with four loss components:

- **Multi-Similarity Loss**: Mines all informative positive/negative pairs per batch
- **Proxy-NCA Loss**: Learns a proxy centroid for each of 8 demographic classes
- **Age-Continuous Soft Contrastive Loss**: Pair weights decay with age distance
- **Auxiliary Age Regression**: Multi-task signal using SmoothL1 loss

Training uses curriculum learning (age thresholds tighten from 36 to 6 months over 100 epochs) and age-balanced sampling.

---

## Hardware Requirements

| Specification | Value |
|---|---|
| MedSigLIP-448 (base model) | ~1.2 GB |
| Custom weights (all agents) | ~24 MB |
| MedGemma 4B (4-bit quantized) | ~3.5 GB |
| **Total VRAM required** | **<6 GB** |
| Inference time per case | ~3 seconds |
| Internet required | **No** |
| Minimum hardware | Consumer GPU (GTX 1660 or equivalent) |

The system is containerizable via Docker with zero external dependencies at inference time. Patient data never leaves the local machine, ensuring GDPR/HIPAA compliance by design.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Regressor as sole predictor | Empirically validated: no ensemble beat the regressor alone on overall MAE |
| Retrieval for explainability only | Atlas MAE (19.13m) too noisy for numeric prediction, but distance correlates with error (r=+0.26) |
| VLM as narrator, not arbiter | VLM deviation is uncorrelated with regressor error (r=-0.005); better used for explanation |
| Demographic-partitioned indices | Skeletal maturation varies by sex and ethnicity; prevents Caucasian atlas bias |
| Output clamping (±12 months) | Safety bound prevents VLM hallucination from corrupting predictions |
| Full-image embeddings for retrieval | Fixed critical domain mismatch (crop queries vs full-image index) that caused 31.56m archivist MAE |
| Age-range-based confidence | Retrieval-agreement confidence was inversely calibrated; age-range MAE is honest and monotonic |

---

## Disclaimer

This tool is for **research and demonstration purposes only**. It is not a medical device and should not be used for clinical decision-making without expert oversight. MedGemma narratives are descriptive, not diagnostic — they require clinical interpretation.

---

## Citation

```bibtex
@misc{chronosmsk2025,
    title={Chronos-MSK: Bias-Aware Skeletal Maturity Assessment at the Edge},
    year={2025},
    note={Google HAI-DEF Competition Submission}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedSigLIP and MedGemma
- [RSNA](https://www.rsna.org/) for the Pediatric Bone Age dataset
- USC for the Digital Hand Atlas
