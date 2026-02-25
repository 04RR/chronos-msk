# Chronos-MSK: Demographically Fair Bone Age Assessment via Multi-Agent Retrieval

## Write up present in ChronosMSK_Writeup.pdf

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

Validated with positive monotonic correlation (r = +0.20):

| Tier | Cases | MAE | Within ±12m |
|---|---|---|---|
| HIGH | 535 (37.5%) | 6.96m | 83.9% |
| MODERATE | 644 (45.2%) | 9.69m | 67.2% |
| LOW | 245 (17.2%) | 10.71m | 64.5% |

### Stratified Performance

**Sex**: Male MAE = 8.26m, Female MAE = 9.47m

| Age Range | Cases | Regressor MAE | Atlas MAE |
|---|---|---|---|
| 0-5 years | 94 | 9.54m | 28.49m |
| 5-10 years | 394 | 10.04m | 31.33m |
| 10-15 years | 809 | 8.07m | 12.68m |
| 15-19 years | 124 | 8.45m | 15.81m |

---

## Architecture

The system orchestrates six specialized agents:

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

1. **Feature extraction**: Frozen encoder produces 1152-D embeddings for classification and retrieval
2. **Regression backbone**: LoRA-adapted with a 4-layer head incorporating sex conditioning for DLDL age estimation
3. **Atlas embedding engine**: Full-image embeddings projected through a trained 256-D metric space for demographic-partitioned FAISS retrieval

The 1.2 GB base model supports all three downstream tasks through lightweight adapters and heads totaling only 24 MB.

### MedGemma 1.5 4B-IT — The Reasoning Layer

Receives structured evidence (regressor estimate, atlas matches with distances, confidence tier) alongside the X-ray image. Generates formal radiology reports with Findings, Impression, and Bone Age Assessment sections. Does not override the quantitative prediction — it explains it.

Validated behavior (1,425 cases):
- Agrees with regressor 76% of the time
- Output clamped to ±12 months as safety bound
- Only 3% of cases required clamping

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with ≥6 GB VRAM (or CPU — slower but functional)
- ~3 GB disk space for model weights and indices
- Internet connection for first run (downloads MedSigLIP-448 from HuggingFace, ~1.2 GB, cached after)

### 1. Clone and Install

```bash
git clone https://github.com/04RR/chronos-msk.git
cd chronos-msk
pip install -r requirements.txt
```

### 2. Download Weights

Download from Kaggle and extract into the project:

```bash
# Option A: Kaggle CLI
pip install kaggle
kaggle datasets download -d rohitrajesh/model-checkpoints -p weights/ --unzip

# Option B: Manual
# Visit https://www.kaggle.com/datasets/rohitrajesh/model-checkpoints
# Download, unzip, and place contents into weights/
```

If `medsiglip_sota` downloads as a zip, unzip it:

```bash
cd weights && unzip medsiglip_sota.zip && cd ..
```

Verify structure:

```
weights/
├── best_scout.pt
├── radiologist_head.pkl
├── projector_sota.pth
└── medsiglip_sota/
    ├── config.json
    ├── heads.pth
    └── adapter/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

### 3. Download Retrieval Index

```bash
# Option A: Kaggle CLI
kaggle datasets download -d rohitrajesh/indices-projected-256d -p . --unzip

# Option B: Manual
# Visit https://www.kaggle.com/datasets/rohitrajesh/indices-projected-256d
# Download, unzip into project root as indices_projected_256d/
```

Verify structure:

```
indices_projected_256d/
├── Male_Asian.index
├── Male_Asian_meta.json
├── ...  (8 .index + 8 _meta.json files)
└── Female_Black_meta.json
```

### 4. Add Sample Images

Place at least one hand/wrist X-ray PNG in a `samples/` directory for the built-in examples. The example filenames expected are `1561.png` and `1970.png` (from the RSNA Bone Age dataset).

```bash
mkdir -p samples
# Copy sample X-rays into samples/
```

If you don't have RSNA images, the app still works — just upload your own image instead of using the examples.

### 5. Launch

```bash
python app.py
```

First launch downloads MedSigLIP-448 from HuggingFace (~1.2 GB). Subsequent launches use the cache.

Navigate to **http://localhost:7860** once you see:

```
✅ All agents loaded. Ready to serve.
```

### 6. Optional: Enable MedGemma Narratives

Clinical narrative generation requires [LM Studio](https://lmstudio.ai/) running locally with the `medgemma-1.5-4b-it` model. This is **entirely optional** — the core bone age pipeline works without it.

1. Install LM Studio
2. Search for and download `medgemma-1.5-4b-it`
3. Start the local server (default: `http://localhost:1234`)
4. Restart `app.py` — it will auto-detect the VLM

If LM Studio is not running, the app disables narrative generation automatically with no errors.

### 6. Run Evaluation

```bash
# Fast evaluation — ensemble only, no VLM (~30 min, requires GPU)
python evaluate.py --mode fast \
    --val-csv data/boneage_val.csv \
    --image-dir data/RSNA_val/images

# Full evaluation with MedGemma narratives (~2 hrs, requires LM Studio)
python evaluate.py --mode full \
    --vlm-url http://localhost:1234/v1/chat/completions

# Compute metrics from existing results
python evaluate.py --mode metrics --results evaluation_results/results.csv
```

---

## Project Structure

```
chronos-msk/
│
├── app.py                              # Gradio demo interface
├── evaluate.py                         # Consolidated evaluation script
├── compute_competition_metrics.py      # Metrics summary generator
│
├── agents/
│   ├── __init__.py
│   ├── agent1_scout.py                 # YOLOv8 radius detection
│   ├── agent2_radiologist.py           # MedSigLIP + SVM staging
│   ├── agent3_archivist.py             # FAISS retrieval with SOTA projector
│   ├── agent4_vlm_client.py            # MedGemma VLM client with output clamping
│   ├── agent5_regressor.py             # MedSigLIP + LoRA regression
│   └── agent6_ensemble.py              # Confidence-calibrated ensemble
│
├── training/
│   ├── train_scout.py                  # YOLO detector training
│   ├── train_radiologist.py            # SVM head training on MedSigLIP embeddings
│   ├── train_regressor.py              # LoRA fine-tuning for DLDL age regression
│   ├── train_retriever_sota.py         # SOTA projector (MS Loss + Proxy-NCA)
│   └── build_full_index.py             # Full-image FAISS index construction
│
└── docs/
    └── writeup.pdf                     # Competition writeup
```

---

## Full Retraining Guide

To reproduce the entire pipeline from scratch:

### Step 1: Train Scout (Agent 1)

Requires annotated bounding box data for distal radius detection.

```bash
python training/train_scout.py
```

### Step 2: Train Radiologist (Agent 2)

Requires TW3 stage labels (we used Gemini 3 Pro as a synthetic annotator).

```bash
python training/train_radiologist.py
```

### Step 3: Train Regressor (Agent 5)

Requires the full RSNA Bone Age dataset (12,611 training images). Optimized for RTX 4090.

```bash
python training/train_regressor.py
```

Key training details:
- **Architecture**: MedSigLIP-448 with DoRA (Weight-Decomposed LoRA), r=16, alpha=32
- **Loss**: KL divergence on Gaussian label distributions (DLDL, sigma=12 months)
- **Optimizer**: AdamW fused, cosine LR schedule, bf16 mixed precision
- **Augmentation**: Rotation (±20°), brightness/contrast jitter
- **Training time**: ~2 hours on RTX 4090

### Step 4: Train Retriever and Build Index (Agent 3)

Requires the Digital Hand Atlas images.

```bash
# This script:
# 1. Scans the atlas directory structure
# 2. Embeds all 1,390 images with MedSigLIP (cached after first run)
# 3. Trains the demographic projector with multi-loss approach
# 4. Rebuilds FAISS indices automatically
python training/train_retriever_sota.py
```

Training uses four loss components:
- **Multi-Similarity Loss**: Mines all informative positive/negative pairs
- **Proxy-NCA Loss**: Learns demographic class proxies
- **Age-Continuous Soft Contrastive Loss**: Smooth age gradients via soft pair weights
- **Auxiliary Age Regression**: Multi-task signal
- **Curriculum Learning**: Age thresholds tighten from 36 to 6 months over 100 epochs

### Step 5: Evaluate

```bash
python evaluate.py --mode fast
```

---

## Datasets

### RSNA Pediatric Bone Age (14,236 images)

Primary training and validation source. We used a held-out 1,425-case validation set with strict no-leakage protocol. Labels are bone age in months with sex metadata.

Source: [Kaggle](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

### USC Digital Hand Atlas (1,390 images)

Explicitly designed for ethnic diversity with even distribution across four racial groups:

| Group | Male | Female | Total |
|---|---|---|---|
| Asian | 167 | 167 | 334 |
| Black | 184 | 174 | 358 |
| Caucasian | 167 | 166 | 333 |
| Hispanic | 182 | 183 | 365 |

FAISS indices are partitioned by Sex and Race (8 partitions), ensuring Visual Twins come from biologically relevant populations.

Source: [USC DHA System](https://ipilab.usc.edu/research/baaweb/)

---

## Hardware Requirements

| Specification | Value |
|---|---|
| MedSigLIP-448 (base model) | ~1.2 GB |
| Custom weights (all agents) | ~24 MB |
| MedGemma 4B (4-bit quantized) | ~3.5 GB |
| FAISS indices | ~3 MB |
| **Total VRAM required** | **< 6 GB** |
| Inference time per case | ~3 seconds |
| Internet required | **No** |
| Minimum hardware | Consumer GPU (GTX 1660 or equivalent) |

For training the regressor: RTX 4090 recommended (~2 hours). RTX 3080+ will also work with reduced batch size.

---

## Key Design Decisions

| Decision | Rationale | Evidence |
|---|---|---|
| Regressor as sole predictor | No ensemble beat the regressor alone on overall MAE | Grid search over all threshold/weight combinations |
| Retrieval for explainability only | Atlas MAE (19.13m) too noisy for numeric prediction | Distance-error correlation r=+0.26 confirms meaningful space |
| VLM as narrator, not arbiter | VLM deviation uncorrelated with regressor error (r=-0.005) | 1,425-case evaluation: VLM agrees 76% of the time |
| Full-image embedding for retrieval | Fixed domain mismatch (crop query vs full-image index) | Archivist MAE improved from 31.56m to 19.13m |
| Demographic-partitioned indices | Skeletal maturation varies by sex and ethnicity | DHA provides explicit race/sex metadata |
| Output clamping (±12 months) | Safety bound prevents VLM hallucination | Pre-fix: VLM outputs ranged 0-1750 months |
| DLDL instead of direct regression | Probability distribution captures uncertainty naturally | Expectation trick provides smooth, stable predictions |

---

## Disclaimer

This tool is for **research and demonstration purposes only**. It is not a medical device and should not be used for clinical decision-making without expert oversight. MedGemma narratives are descriptive, not diagnostic — they require clinical interpretation.

---

## Citation

```bibtex
@misc{chronosmsk2025,
    title={Chronos-MSK: Bias-Aware Skeletal Maturity Assessment at the Edge},
    author={Rohit Rajesh},
    year={2025},
    note={Google HAI-DEF Competition Submission}
}
```

---

## Acknowledgments

- [Google Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations) for MedSigLIP and MedGemma
- [RSNA](https://www.rsna.org/) for the Pediatric Bone Age dataset
- USC for the Digital Hand Atlas
- [razorx89](https://github.com/razorx89/digital-hand-atlas-downloader) for the DHA downloader

---

## License

MIT License. See [LICENSE](LICENSE) for details.
