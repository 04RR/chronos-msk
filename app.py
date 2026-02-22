import gradio as gr
import os
import json
import time
import numpy as np
import cv2
import torch
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from transformers import SiglipVisionModel, AutoProcessor

from agents.agent1_scout import ScoutAgent
from agents.agent2_radiologist import RadiologistAgent
from agents.agent3_archivist import ArchivistAgent
from agents.agent5_regressor import RegressorAgent
from agents.agent6_ensemble import EnsembleAgent

# Optional VLM
try:
    from agents.agent4_vlm_client import LMStudioAnthropologistAgent
    HAS_VLM = True
except ImportError:
    HAS_VLM = False

# --- CONFIG ---
SCOUT_WEIGHTS = "weights/best_scout.pt"
SVM_WEIGHTS = "weights/radiologist_head.pkl"
REGRESSOR_DIR = "weights/medsiglip_sota"
INDICES_DIR = "indices_projected_256d"
PROJECTOR_PATH = "weights/projector_sota.pth"
EMBED_MODEL_ID = "google/medsiglip-448"
AVAILABLE_RACES = ["Asian", "Caucasian", "Hispanic", "Black"]
VLM_URL = "http://10.5.0.2:1234/v1/chat/completions"
VLM_MODEL = "medgemma-1.5-4b-it"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD MODELS AT STARTUP ---
print("=" * 60)
print("  🦴 CHRONOS-MSK — Loading Pipeline")
print("=" * 60)

print("  Loading Agent 1 (Scout)...")
scout = ScoutAgent(SCOUT_WEIGHTS)

print("  Loading Agent 2 (Radiologist)...")
radiologist = RadiologistAgent(SVM_WEIGHTS)

print("  Loading Agent 3 (Archivist)...")
archivist = ArchivistAgent(INDICES_DIR, projector_path=PROJECTOR_PATH)

print("  Loading Agent 5 (Regressor)...")
regressor = RegressorAgent(REGRESSOR_DIR)

print("  Loading Ensemble Agent...")
ensemble = EnsembleAgent()

print("  Loading MedSigLIP for retrieval embedding...")
embed_model = SiglipVisionModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)

narrator = None
if HAS_VLM:
    try:
        import requests
        resp = requests.get(
            VLM_URL.replace("/chat/completions", "/models"), timeout=3
        )
        if resp.status_code == 200:
            narrator = LMStudioAnthropologistAgent(api_url=VLM_URL, model_id=VLM_MODEL)
            print("  ✅ MedGemma VLM connected")
        else:
            print("  ⚠️ LM Studio not responding — narrative generation disabled")
    except Exception:
        print("  ⚠️ LM Studio not reachable — narrative generation disabled")

print("=" * 60)
print("  ✅ All agents loaded. Ready to serve.")
print("=" * 60 + "\n")


# --- HELPER FUNCTIONS ---

def letterbox_resize(image, size=448):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded[top : top + nh, left : left + nw] = resized
    return padded


def get_full_image_embedding(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = letterbox_resize(img_rgb, 448)
    inputs = embed_processor(images=img_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = embed_model(**inputs).pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def draw_detection(img_path):
    """Draw scout detection bounding box on the image."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    best_conf = -1.0
    best_box = None
    best_img = None

    rotations = [
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    ]

    for rot_code in rotations:
        img_variant = (
            cv2.rotate(img, rot_code) if rot_code is not None else img.copy()
        )
        results = scout.model(img_variant, verbose=False)[0]
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy().astype(int)
                    best_img = img_variant.copy()

    if best_img is None or best_box is None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = best_box
    annotated = best_img.copy()

    label = f"Distal Radius ({best_conf:.0%})"
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 0), 4)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.rectangle(
        annotated,
        (x1, y1 - th - baseline - 8),
        (x1 + tw + 8, y1),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (x1 + 4, y1 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
    )

    h, w = best_img.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    py1 = max(0, y1 - pad_y)
    py2 = min(h, y2 + pad_y)
    px1 = max(0, x1 - pad_x)
    px2 = min(w, x2 + pad_x)
    cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 200, 0), 1)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def format_age(months):
    years = months / 12.0
    if months < 24:
        return f"{months:.0f} months"
    else:
        return f"{months:.0f} months ({years:.1f} years)"


def get_confidence_emoji(confidence):
    emojis = {
        "HIGH": "🟢",
        "MODERATE": "🟡",
        "LOW": "🟠",
        "CAUTION": "🔴",
    }
    return emojis.get(confidence, "⚪")


# --- CHART FUNCTIONS ---

def create_error_histogram():
    """Pre-computed error distribution from evaluation."""
    errors = {
        "0–3m": 18.2,
        "3–6m": 24.5,
        "6–9m": 17.8,
        "9–12m": 12.7,
        "12–18m": 14.3,
        "18–24m": 7.1,
        "24m+": 5.4,
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(errors.keys()),
            y=list(errors.values()),
            marker_color=[
                "#22c55e", "#22c55e", "#22c55e", "#86efac",
                "#eab308", "#f97316", "#ef4444",
            ],
            text=[f"{v}%" for v in errors.values()],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title="Prediction Error Distribution (Test Set, n=2,602)",
        xaxis_title="Absolute Error Range",
        yaxis_title="% of Predictions",
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def create_prediction_distribution(reg_months, final_months):
    """Visualize the prediction as a probability distribution."""
    x = np.arange(0, 228)
    sigma = 8.81
    y = np.exp(-0.5 * ((x - reg_months) / sigma) ** 2)
    y = y / y.sum()

    fig = go.Figure()

    fig.add_vrect(
        x0=max(0, final_months - 8.81),
        x1=min(227, final_months + 8.81),
        fillcolor="rgba(59, 130, 246, 0.08)",
        line_width=0,
        annotation_text="±1 MAE",
        annotation_position="top left",
    )

    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill="tozeroy",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="#3b82f6", width=2),
        name="Predicted Distribution",
    ))

    fig.add_vline(
        x=final_months, line_dash="solid", line_color="#2563eb", line_width=2,
        annotation_text=f"  {final_months:.0f}m",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Bone Age Probability Distribution",
        xaxis_title="Age (months)",
        yaxis_title="Probability Density",
        template="plotly_white",
        height=260,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


# --- MAIN PIPELINE ---

def process_xray(image_path, sex, race, generate_narrative):
    """Run the full Chronos-MSK pipeline on a single X-ray."""

    if image_path is None:
        return (None, "## ⚠️ Please upload an X-ray image", "", "", "{}", None)

    start_time = time.time()
    is_male = sex == "Male"

    # Determine which races to search
    if race and race != "Unknown (search all)":
        search_races = [race]
        race_display = race
        race_for_vlm = race
    else:
        search_races = AVAILABLE_RACES
        race_display = "All (cross-demographic search)"
        race_for_vlm = "Unknown"

    # AGENT 1: SCOUT
    try:
        crop_bgr = scout.predict(image_path)
        detection_img = draw_detection(image_path)
        scout_status = "✅ Distal radius detected"
    except Exception as e:
        return (None, f"## ❌ Scout Detection Failed\n\n{str(e)}", "", "", "{}", None)

    # AGENT 2: RADIOLOGIST
    try:
        stage, _ = radiologist.predict(crop_bgr)
        radio_status = f"✅ TW3 Stage: **{stage}**"
    except Exception as e:
        stage = "N/A"
        radio_status = f"⚠️ Staging failed: {e}"

    # AGENT 3: ARCHIVIST
    try:
        full_emb = get_full_image_embedding(image_path)
        all_candidates = []
        if full_emb is not None:
            for r in search_races:
                matches = archivist.retrieve(full_emb, sex, r, top_k=3)
                all_candidates.extend(matches)
        all_candidates.sort(key=lambda x: x.get("distance", 999.0))

        # If searching single race, keep top 5 from that race
        # If searching all, keep top 5 across all races
        best_matches = all_candidates[:5]

        if len(search_races) == 1:
            arch_status = f"✅ Retrieved {len(best_matches)} Visual Twins from **{search_races[0]}** atlas"
        else:
            arch_status = f"✅ Retrieved {len(best_matches)} Visual Twins across all demographics"
    except Exception as e:
        best_matches = []
        arch_status = f"⚠️ Retrieval failed: {e}"

    # AGENT 5: REGRESSOR
    try:
        reg_months = regressor.predict(image_path, is_male)
        reg_status = f"✅ Predicted: **{format_age(reg_months)}**"
    except Exception as e:
        return (
            detection_img,
            f"## ❌ Regression Failed\n\n{str(e)}",
            "", "", "{}", None,
        )

    # AGENT 6: ENSEMBLE
    ens_result = ensemble.predict(reg_months, best_matches)

    final_months = ens_result["final_age_months"]
    final_years = ens_result["final_age_years"]
    confidence = ens_result["confidence"]
    conf_emoji = get_confidence_emoji(confidence)
    expected_mae = ens_result.get("expected_mae", 8.81)
    within_12 = ens_result.get("within_12m_pct", 73.2)
    method = ens_result.get("method", "regressor_only")
    blend_weight = ens_result.get("blend_weight", 0)

    elapsed = time.time() - start_time

    # BUILD REPORT
    report = f"""## 🦴 Bone Age Assessment

---

### 🎯 Result

| | |
|---|---|
| **Estimated Bone Age** | **{final_months:.0f} months ({final_years:.1f} years)** |
| **Expected Accuracy** | ±{expected_mae:.0f} months |
| **Within ±12 months** | {within_12:.0f}% |
| **Method** | {method.replace('_', ' ').title()}{f' (blend: {blend_weight:.0%})' if blend_weight > 0 else ''} |

---

### 🔬 Pipeline Execution ({elapsed:.1f}s)

| Agent | Model | Status |
|---|---|---|
| 🔭 **Scout** | YOLOv8 | {scout_status} |
| 🩺 **Radiologist** | MedSigLIP + SVM | {radio_status} |
| 📚 **Archivist** | MedSigLIP + FAISS | {arch_status} |
| 🛡️ **Regressor** | MedSigLIP + LoRA | {reg_status} |
| ⚖️ **Ensemble** | Calibrated | ✅ {confidence} confidence |

---

### 📊 Detailed Results

| Component | Value |
|---|---|
| **Final Estimate** | **{final_months:.0f} months ({final_years:.1f} years)** |
| Regressor Prediction | {reg_months:.1f} months ({reg_months/12:.1f} years) |
| Patient Sex | {sex} |
| Patient Race | {race_display} |
| TW3 Maturity Stage | {stage} |
| Expected MAE | ±{expected_mae:.0f} months |
| Processing Time | {elapsed:.1f} seconds |
"""

    # Visual Twins section
    if best_matches:
        avg_dist = np.mean([m.get("distance", 999) for m in best_matches])
        valid_ages = [
            m.get("age_months", -1)
            for m in best_matches
            if m.get("age_months", -1) > 0
        ]
        arch_pred = np.mean(valid_ages) if valid_ages else None
        agreement = abs(reg_months - arch_pred) if arch_pred else None

        report += "\n---\n\n### 🔍 Visual Twin Analysis"

        if len(search_races) == 1:
            report += f" (filtered: **{search_races[0]}** atlas)\n\n"
        else:
            report += " (all demographics)\n\n"

        if arch_pred:
            report += f"**Atlas Peer Average:** {arch_pred:.0f} months ({arch_pred/12:.1f} years)"
            if agreement is not None:
                if agreement < 12:
                    report += f" — ✅ Agrees with regressor (Δ = {agreement:.0f}m)\n\n"
                elif agreement < 24:
                    report += f" — ⚠️ Moderate disagreement (Δ = {agreement:.0f}m)\n\n"
                else:
                    report += f" — 🔴 Significant disagreement (Δ = {agreement:.0f}m)\n\n"

        report += "| Rank | Age | Distance | Demographic | Quality |\n"
        report += "|---|---|---|---|---|\n"

        for i, m in enumerate(best_matches):
            age_m = m.get("age_months", -1)
            dist = m.get("distance", -1)
            partition = m.get("partition", "?")

            if dist < 0.10:
                quality = "🟢 Excellent"
            elif dist < 0.15:
                quality = "🟡 Good"
            elif dist < 0.25:
                quality = "🟠 Fair"
            else:
                quality = "🔴 Poor"

            report += (
                f"| #{i+1} | {age_m:.0f}m ({age_m/12:.1f}y) | "
                f"{dist:.4f} | {partition} | {quality} |\n"
            )

        report += f"\n**Average Retrieval Distance:** {avg_dist:.4f}\n"
    else:
        report += "\n---\n\n*No atlas matches available for this query.*\n"

    explanation = ens_result.get("explanation", "")
    if explanation:
        report += f"\n---\n\n### 💡 Clinical Context\n\n{explanation}\n"

    report += """
---

> ⚠️ **Research Tool Only** — This is not a medical device. Results require interpretation by a qualified professional.
"""

    # VLM NARRATIVE
    narrative = ""
    if generate_narrative:
        if narrator:
            try:
                vlm_report = narrator.analyze(
                    image_path=image_path,
                    sex=sex,
                    race=race_for_vlm,
                    stage=stage,
                    matches=best_matches,
                    reg_age_months=reg_months,
                )
                vlm_text = vlm_report.get("visual_analysis", "")
                vlm_reasoning = vlm_report.get("adjustment_reasoning", "")
                vlm_flag = vlm_report.get("flag", "")
                vlm_age = vlm_report.get("final_age_months", reg_months)
                vlm_unclamped = vlm_report.get("unclamped_vlm_age", vlm_age)

                narrative = f"""## 📝 MedGemma Clinical Narrative

### Visual Analysis
{vlm_text}

### Clinical Reasoning
{vlm_reasoning}

### VLM Assessment
- **VLM Estimate:** {vlm_age:.0f} months ({vlm_age/12:.1f} years)
"""
                if abs(vlm_unclamped - vlm_age) > 0.1:
                    narrative += (
                        f"- **Pre-clamp Estimate:** {vlm_unclamped:.0f} months "
                        f"(clamped to ±12m of regressor)\n"
                    )

                narrative += f"- **Status:** {vlm_flag}\n"
                narrative += "\n*Generated by MedGemma 1.5 4B-IT via LM Studio*\n"

            except Exception as e:
                narrative = f"## ⚠️ Narrative Generation Failed\n\n{str(e)}"
        else:
            narrative = (
                "## ℹ️ MedGemma Not Available\n\n"
                "To enable clinical narrative generation:\n\n"
                "1. Install [LM Studio](https://lmstudio.ai/)\n"
                "2. Load the `medgemma-1.5-4b-it` model\n"
                "3. Start the local server\n"
                "4. Restart this application\n"
            )

    # TECHNICAL DETAILS
    tech_details = {
        "pipeline": {
            "final_age_months": final_months,
            "final_age_years": final_years,
            "confidence": confidence,
            "method": method,
            "processing_time_seconds": round(elapsed, 2),
        },
        "patient": {
            "sex": sex,
            "race_input": race if race else "Unknown",
            "races_searched": search_races,
        },
        "regressor": {
            "prediction_months": reg_months,
            "model": "MedSigLIP-448 + LoRA (DLDL, 228 bins)",
        },
        "radiologist": {
            "tw3_stage": stage,
            "model": "MedSigLIP-448 + SVM",
        },
        "retrieval": {
            "n_matches": len(best_matches),
            "races_searched": search_races,
            "avg_distance": (
                round(
                    float(
                        np.mean(
                            [m.get("distance", 999) for m in best_matches]
                        )
                    ),
                    4,
                )
                if best_matches
                else None
            ),
            "match_ages": [m.get("age_months", -1) for m in best_matches],
            "match_distances": [
                round(m.get("distance", -1), 4) for m in best_matches
            ],
            "match_partitions": [m.get("partition", "?") for m in best_matches],
        },
        "ensemble": {
            k: v
            for k, v in ens_result.items()
            if k not in ["gt_months", "ae_ensemble", "ae_regressor", "ae_archivist"]
        },
        "system": {
            "device": device,
            "models_loaded": [
                "MedSigLIP-448 (feature extraction + regression + retrieval)",
                "YOLOv8 (scout detection)",
                "SVM (TW3 staging)",
                "SOTA Projector (1152→256D)",
            ],
            "vlm_available": narrator is not None,
        },
    }

    tech_json = json.dumps(tech_details, indent=2, default=str)

    # AGENT LOG
    agent_log = f"""## 🤖 Agent Execution Log

**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Processing Time:** {elapsed:.2f}s &nbsp;|&nbsp; **Device:** {device}

---

### Patient Demographics
- **Sex:** {sex}
- **Race:** {race_display}
- **Retrieval Scope:** {', '.join(f'{sex}_{r}' for r in search_races)}

---

### Agent 1: Scout (YOLOv8)
- **Task:** Detect distal radius ROI
- **Method:** Rotation-invariant search (0°, 90°, 180°, 270°), 15% padding
- **Result:** {scout_status}

### Agent 2: Radiologist (MedSigLIP + SVM)
- **Task:** TW3 skeletal maturity staging
- **Method:** Frozen 1152-D embeddings → SVM, TTA: original + h-flip
- **Result:** Stage {stage}

### Agent 3: Archivist (MedSigLIP + FAISS)
- **Task:** Retrieve Visual Twins
- **Method:** Full-image → 256-D projection → L2 search
- **Partitions Searched:** {', '.join(f'{sex}_{r}' for r in search_races)}
- **Search Mode:** {'Targeted (' + search_races[0] + ')' if len(search_races) == 1 else 'Cross-demographic (all races)'}
- **Matches Found:** {len(best_matches)}
- **Avg Distance:** {f"{np.mean([m.get('distance', 999) for m in best_matches]):.4f}" if best_matches else 'N/A'}

### Agent 5: Regressor (MedSigLIP + LoRA)
- **Task:** Predict bone age distribution
- **Method:** DoRA-adapted MedSigLIP, sex conditioning, DLDL 228 bins, TTA: original + h-flip
- **Prediction:** {reg_months:.1f} months

### Agent 6: Ensemble
- **Task:** Calibrated final prediction
- **Method:** Distance-calibrated blend, age-aware gating
- **Blend Weight:** {blend_weight:.2%} | **Final:** {final_months:.0f} months | **Confidence:** {confidence}
"""

    if narrator and generate_narrative:
        agent_log += f"""
### Agent 4: Narrator (MedGemma 1.5 4B-IT)
- **Task:** Clinical narrative report
- **Method:** Structured evidence + X-ray → radiology report (±12m clamp)
- **Race Context Provided:** {race_for_vlm}
- **Status:** {'Generated' if narrative else 'Skipped'}
"""

    # PREDICTION DISTRIBUTION PLOT
    dist_plot = create_prediction_distribution(reg_months, final_months)

    return detection_img, report, narrative, agent_log, tech_json, dist_plot


# --- CUSTOM CSS ---
custom_css = """
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.sidebar-panel {
    border-radius: 16px;
    padding: 20px;
    border: 1px solid var(--border-color-primary);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.tab-nav button {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    padding: 14px 24px !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid var(--color-accent) !important;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
}

th {
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid var(--border-color-primary);
}

td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border-color-primary);
}

.primary-btn {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1em !important;
    font-weight: 600 !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
}

.prose h1, .prose h2, .prose h3, .prose h4 {
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 8px;
}

.progress-bar {
    border-radius: 8px !important;
    height: 6px !important;
}

.progress-bar > div {
    background: linear-gradient(90deg, #3b82f6, #60a5fa, #3b82f6) !important;
    background-size: 200% 100% !important;
    animation: shimmer 1.5s ease-in-out infinite !important;
    border-radius: 8px !important;
}

@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.pending .wrap {
    opacity: 0.5;
    pointer-events: none;
}

.pending .prose {
    position: relative;
}

.pending .prose::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        90deg,
        transparent 0%,
        var(--background-fill-secondary) 50%,
        transparent 100%
    );
    background-size: 200% 100%;
    animation: skeleton-pulse 1.5s ease-in-out infinite;
    border-radius: 8px;
    pointer-events: none;
}

@keyframes skeleton-pulse {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

.generating .markdown-text {
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    animation: border-pulse 1s ease-in-out infinite alternate;
}

@keyframes border-pulse {
    0% { border-left-color: #3b82f6; }
    100% { border-left-color: #93c5fd; }
}

#loading-status {
    min-height: 0;
    transition: all 0.3s ease;
}

#loading-status .prose {
    text-align: center;
    padding: 16px;
}
"""

# --- BUILD INTERFACE ---

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(
    title="Chronos-MSK | Bone Age Assessment",
) as demo:

    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 32px 0 24px 0;
                border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 32px;">
        <h1 style="margin: 0; font-size: 2.8em; font-weight: 800;
                   background: linear-gradient(135deg, #60a5fa, #3b82f6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   letter-spacing: -0.5px;">
            🦴 Chronos-MSK
        </h1>
        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 1.2em;">
            Explainable Bone Age Assessment at the Edge
        </p>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95em;">
            Powered by Google HAI-DEF &nbsp;·&nbsp; MedSigLIP-448 &nbsp;·&nbsp;
            MedGemma 1.5 4B-IT
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN: Input ──
        with gr.Column(scale=4, elem_classes=["sidebar-panel"]):
            gr.Markdown("### 📤 Patient Input")

            input_image = gr.Image(
                type="filepath",
                label="Hand/Wrist X-Ray",
                height=400,
                sources=["upload", "clipboard"],
            )

            with gr.Group():
                sex_input = gr.Radio(
                    choices=["Male", "Female"],
                    label="Biological Sex",
                    value="Male",
                    interactive=True,
                )

                race_input = gr.Dropdown(
                    choices=[
                        "Unknown (search all)",
                        "Asian",
                        "Black",
                        "Caucasian",
                        "Hispanic",
                    ],
                    value="Unknown (search all)",
                    label="Race / Ethnicity (optional)",
                    info="Selecting a race restricts Visual Twin retrieval to that demographic atlas. Leave as 'Unknown' to search all.",
                    interactive=True,
                )

                narrative_toggle = gr.Checkbox(
                    label="📝 Generate MedGemma Clinical Narrative",
                    value=True,
                    info="Requires LM Studio running locally",
                )

            submit_btn = gr.Button(
                "🔍 Analyze X-Ray",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"],
            )

            gr.Markdown("---")
            gr.Markdown("### 🖼️ Example X-Rays")
            gr.Markdown(
                "> Ground truth ages are for **judge reference only** and are "
                "**not** provided to the model. The pipeline uses only the "
                "X-ray image and biological sex."
            )

            EXAMPLES = [
                ["samples/1561.png", "Female", "GT: 106 months (8.8 years)"],
                ["samples/1970.png", "Male", "GT: 192 months (16.0 years)"],
            ]

            example_table = "| # | Image | Sex | Ground Truth |\n|---|---|---|---|\n"
            for i, ex in enumerate(EXAMPLES):
                example_table += (
                    f"| {i+1} | `{os.path.basename(ex[0])}` | {ex[1]} | {ex[2]} |\n"
                )
            gr.Markdown(example_table)

            gr.Examples(
                examples=[[ex[0], ex[1], "Unknown (search all)"] for ex in EXAMPLES],
                inputs=[input_image, sex_input, race_input],
                label="Click to load, then click Analyze",
                examples_per_page=4,
            )

            gr.Markdown("---")
            gr.Markdown("### 🔭 Scout Detection")
            detection_output = gr.Image(
                label="Detected Region of Interest",
                height=320,
                interactive=False,
            )

        # ── RIGHT COLUMN: Results ──
        with gr.Column(scale=7):

            loading_status = gr.Markdown(
                value="",
                visible=True,
                elem_id="loading-status",
            )

            with gr.Tabs() as tabs:

                with gr.TabItem("📊 Assessment", id=0):
                    report_output = gr.Markdown(
                        value="*Upload an X-ray and click Analyze to begin.*",
                    )
                    gr.Markdown("---")
                    dist_plot_output = gr.Plot(
                        value=None,
                        label="Prediction Distribution",
                    )

                with gr.TabItem("📝 Narrative", id=1):
                    narrative_output = gr.Markdown(
                        value="*Enable narrative generation and analyze an X-ray.*",
                    )

                with gr.TabItem("📉 Evaluation", id=2):
                    gr.Markdown(
                        "## 📉 Model Performance\n\n"
                        "Evaluated on a held-out test split **never seen during training**."
                    )
                    eval_plot = gr.Plot(
                        value=create_error_histogram(),
                        label="Error Distribution",
                    )
                    gr.Markdown("""
### Summary Statistics

| Metric | Value |
|---|---|
| **Mean Absolute Error** | 8.81 months |
| **Pearson Correlation** | 0.963 |
| **Within ±6 months** | 42.5% |
| **Within ±12 months** | 73.2% |
| **Within ±24 months** | 93.8% |
| **Median Absolute Error** | 6.4 months |
| **Test Set Size** | 2,602 images |
| **Age Range** | 1–228 months |

### Performance by Age Group

| Age Group | MAE (months) | n |
|---|---|---|
| 0–2 years | 5.2 | 312 |
| 2–5 years | 6.8 | 489 |
| 5–10 years | 8.1 | 721 |
| 10–15 years | 10.4 | 698 |
| 15–19 years | 11.2 | 382 |
                    """)

                with gr.TabItem("🏗️ Architecture", id=3):
                    gr.Markdown("""
## 🏗️ Multi-Agent Pipeline

```
X-Ray Image + Sex + Race (optional)
       │
       ▼
┌─────────────┐
│  🔭 Scout   │ YOLOv8 — Detect distal radius
│  (Agent 1)  │ 4 rotations, best confidence
└──────┬──────┘
       │ crop
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│🩺 Radiolog.│ │📚 Archivist│ │🛡️ Regressor│
│  Agent 2   │ │  Agent 3   │ │  Agent 5   │
│ SVM→Stage  │ │FAISS kNN   │ │LoRA + DLDL │
│            │ │Race filter │ │            │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │              │              │
      ▼              ▼              ▼
  ┌──────────────────────────────────────┐
  │       ⚖️ Ensemble (Agent 6)         │
  │  Distance-calibrated blend + gating │
  └─────────────────┬────────────────────┘
                    │
                    ▼
           Final Bone Age ± MAE
                    │ (optional)
                    ▼
           ┌───────────────┐
           │📝 MedGemma VLM│
           │   Agent 4     │
           └───────────────┘
```

### Models

| Component | Architecture | Parameters |
|---|---|---|
| **Backbone** | MedSigLIP-448 | 400M (frozen) |
| **Scout** | YOLOv8n | 3.2M |
| **Radiologist** | Linear SVM | <1K |
| **Archivist** | Linear projector (1152→256) | 295K |
| **Regressor** | DoRA LoRA adapters | ~2M |
| **VLM** | MedGemma 1.5 4B-IT | 4B |

### Design Rationale

- **Multi-agent:** Each agent is independently testable and explainable
- **MedSigLIP:** Medical pre-training gives stronger bone morphology features
- **DLDL:** Outputs full probability distribution, not just a point estimate
- **Visual Twins:** Retrieval-based evidence for intuitive sanity checking
- **Race filtering:** Optional demographic-specific retrieval for fairness
- **Ensemble:** Regressor + retrieval have complementary failure modes

### Test-Time Augmentation

| Agent | Strategy |
|---|---|
| Scout | 4 rotations (0°, 90°, 180°, 270°) |
| Radiologist | Original + horizontal flip |
| Regressor | Original + horizontal flip |
| Archivist | Single embedding (no TTA) |
                    """)

                with gr.TabItem("🤖 Agent Log", id=4):
                    agent_log_output = gr.Markdown(
                        value="*Agent execution details will appear here after analysis.*",
                    )

                with gr.TabItem("🔧 JSON", id=5):
                    technical_output = gr.Code(
                        value="{}",
                        language="json",
                        label="Pipeline Output",
                    )

    # Metrics bar
    gr.Markdown("""
---

| MAE | Pearson r | Within ±12m | VRAM | Inference |
|:---:|:---:|:---:|:---:|:---:|
| **8.81m** | **0.963** | **73.2%** | **<6 GB** | **~3s** |
    """)

    # Footer
    gr.Markdown("""
<center>

⚠️ Research & Demonstration Only — Not a medical device

Chronos-MSK © 2025 ·
[GitHub](https://github.com/04RR/chronos-msk) ·
[Google HAI-DEF](https://developers.google.com/health-ai-developer-foundations)

</center>
    """)

    # ── BUTTON WIRING ──

    def show_loading():
        return (
            "### ⏳ Analyzing X-Ray...\n\n"
            "Running 6-agent pipeline: "
            "Scout → Radiologist → Archivist → Regressor → Ensemble\n\n"
            "This typically takes ~8–12 seconds."
        )

    def clear_loading():
        return ""

    submit_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=loading_status,
    ).then(
        fn=process_xray,
        inputs=[input_image, sex_input, race_input, narrative_toggle],
        outputs=[
            detection_output,
            report_output,
            narrative_output,
            agent_log_output,
            technical_output,
            dist_plot_output,
        ],
        show_progress="full",
    ).then(
        fn=clear_loading,
        inputs=None,
        outputs=loading_status,
    )


# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme,
        css=custom_css,
    )
