import gradio as gr
import os
import json
import time
import numpy as np
import cv2
import torch
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
    """Format age in months to a readable string."""
    years = months / 12.0
    if months < 24:
        return f"{months:.0f} months"
    else:
        return f"{months:.0f} months ({years:.1f} years)"


def get_confidence_color(confidence):
    """Return color for confidence tier."""
    colors = {
        "HIGH": "#22c55e",
        "MODERATE": "#eab308",
        "LOW": "#f97316",
        "CAUTION": "#ef4444",
    }
    return colors.get(confidence, "#6b7280")


def get_confidence_emoji(confidence):
    emojis = {
        "HIGH": "🟢",
        "MODERATE": "🟡",
        "LOW": "🟠",
        "CAUTION": "🔴",
    }
    return emojis.get(confidence, "⚪")


# --- MAIN PIPELINE ---

def process_xray(image_path, sex, generate_narrative):
    """Run the full Chronos-MSK pipeline on a single X-ray."""

    if image_path is None:
        return (
            None,
            "## ⚠️ Please upload an X-ray image",
            "",
            "",
            "{}",
        )

    start_time = time.time()
    is_male = sex == "Male"

    # ================================================================
    # AGENT 1: SCOUT — Anatomical Detection
    # ================================================================
    try:
        crop_bgr = scout.predict(image_path)
        detection_img = draw_detection(image_path)
        scout_status = "✅ Distal radius detected"
    except Exception as e:
        return (
            None,
            f"## ❌ Scout Detection Failed\n\n{str(e)}",
            "",
            "",
            "{}",
        )

    # ================================================================
    # AGENT 2: RADIOLOGIST — TW3 Staging
    # ================================================================
    try:
        stage, _ = radiologist.predict(crop_bgr)
        radio_status = f"✅ TW3 Stage: **{stage}**"
    except Exception as e:
        stage = "N/A"
        radio_status = f"⚠️ Staging failed: {e}"

    # ================================================================
    # AGENT 3: ARCHIVIST — Visual Twin Retrieval
    # ================================================================
    try:
        full_emb = get_full_image_embedding(image_path)
        all_candidates = []
        if full_emb is not None:
            for race in AVAILABLE_RACES:
                matches = archivist.retrieve(full_emb, sex, race, top_k=3)
                all_candidates.extend(matches)
        all_candidates.sort(key=lambda x: x.get("distance", 999.0))
        best_matches = all_candidates[:5]
        arch_status = f"✅ Retrieved {len(best_matches)} Visual Twins"
    except Exception as e:
        best_matches = []
        arch_status = f"⚠️ Retrieval failed: {e}"

    # ================================================================
    # AGENT 5: REGRESSOR — Age Prediction
    # ================================================================
    try:
        reg_months = regressor.predict(image_path, is_male)
        reg_status = f"✅ Predicted: **{format_age(reg_months)}**"
    except Exception as e:
        return (
            detection_img,
            f"## ❌ Regression Failed\n\n{str(e)}",
            "",
            "",
            "{}",
        )

    # ================================================================
    # AGENT 6: ENSEMBLE — Calibrated Prediction
    # ================================================================
    ens_result = ensemble.predict(reg_months, best_matches)

    final_months = ens_result["final_age_months"]
    final_years = ens_result["final_age_years"]
    confidence = ens_result["confidence"]
    conf_emoji = get_confidence_emoji(confidence)
    conf_color = get_confidence_color(confidence)
    expected_mae = ens_result.get("expected_mae", 8.81)
    within_12 = ens_result.get("within_12m_pct", 73.2)
    method = ens_result.get("method", "regressor_only")
    blend_weight = ens_result.get("blend_weight", 0)

    elapsed = time.time() - start_time

    # ================================================================
    # BUILD ASSESSMENT REPORT (Pure Markdown — no inline HTML)
    # ================================================================

    report = f"""## 🦴 Bone Age Assessment
### Chronos-MSK Multi-Agent Pipeline

---

### 🎯 Result

| | |
|---|---|
| **Estimated Bone Age** | **{final_months:.0f} months ({final_years:.1f} years)** |
| **Confidence** | {conf_emoji} {confidence} |
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
| TW3 Maturity Stage | {stage} |
| Confidence Tier | {conf_emoji} {confidence} |
| Expected MAE | ±{expected_mae:.0f} months |
| Processing Time | {elapsed:.1f} seconds |
"""

    # Visual Twins section
    if best_matches:
        avg_dist = np.mean([m.get("distance", 999) for m in best_matches])
        valid_ages = [m.get("age_months", -1) for m in best_matches if m.get("age_months", -1) > 0]
        arch_pred = np.mean(valid_ages) if valid_ages else None
        agreement = abs(reg_months - arch_pred) if arch_pred else None

        report += """
---

### 🔍 Visual Twin Analysis (Atlas Retrieval)

"""
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

    # Explanation
    explanation = ens_result.get("explanation", "")
    if explanation:
        report += f"""
---

### 💡 Clinical Context

{explanation}
"""

    # Disclaimer — pure Markdown
    report += """
---

> ⚠️ **Research Tool Only** — This is not a medical device. Results require clinical interpretation by a qualified professional. Powered by Google HAI-DEF (MedSigLIP-448 + MedGemma 1.5 4B-IT).
"""

    # ================================================================
    # VLM NARRATIVE (Optional)
    # ================================================================
    narrative = ""
    if generate_narrative:
        if narrator:
            try:
                vlm_report = narrator.analyze(
                    image_path=image_path,
                    sex=sex,
                    race="Unknown",
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
                    narrative += f"- **Pre-clamp Estimate:** {vlm_unclamped:.0f} months (clamped to ±12m of regressor)\n"

                narrative += f"- **Status:** {vlm_flag}\n"
                narrative += f"\n*Generated by MedGemma 1.5 4B-IT via LM Studio*\n"

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

    # ================================================================
    # TECHNICAL DETAILS (JSON)
    # ================================================================
    tech_details = {
        "pipeline": {
            "final_age_months": final_months,
            "final_age_years": final_years,
            "confidence": confidence,
            "method": method,
            "processing_time_seconds": round(elapsed, 2),
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
            "avg_distance": round(float(np.mean([m.get("distance", 999) for m in best_matches])), 4) if best_matches else None,
            "match_ages": [m.get("age_months", -1) for m in best_matches],
            "match_distances": [round(m.get("distance", -1), 4) for m in best_matches],
            "match_partitions": [m.get("partition", "?") for m in best_matches],
        },
        "ensemble": {
            k: v for k, v in ens_result.items()
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

    # Agent log — pure Markdown
    agent_log = f"""## 🤖 Agent Execution Log

**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Processing Time:** {elapsed:.2f} seconds
**Device:** {device}

---

### Agent 1: Scout (YOLOv8)
- **Task:** Detect distal radius region in X-ray
- **Method:** Rotation-invariant search (0°, 90°, 180°, 270°)
- **Padding:** 15% bounding box expansion
- **Result:** {scout_status}

### Agent 2: Radiologist (MedSigLIP + SVM)
- **Task:** TW3 skeletal maturity staging
- **Method:** Frozen MedSigLIP embeddings (1152-D) → SVM classifier
- **TTA:** Original + horizontal flip, averaged probabilities
- **Result:** Stage {stage}

### Agent 3: Archivist (MedSigLIP + FAISS)
- **Task:** Retrieve demographically-matched "Visual Twins"
- **Method:** Full-image embedding → 256-D SOTA projection → L2 search
- **Partitions Searched:** {', '.join(f'{sex}_{r}' for r in AVAILABLE_RACES)}
- **Matches Found:** {len(best_matches)}
- **Avg Distance:** {f"{np.mean([m.get('distance', 999) for m in best_matches]):.4f}" if best_matches else 'N/A'}

### Agent 5: Regressor (MedSigLIP + LoRA)
- **Task:** Predict bone age as probability distribution
- **Method:** DoRA-adapted MedSigLIP with sex conditioning
- **Output:** Softmax over 228 bins → expected value
- **TTA:** Original + horizontal flip, averaged
- **Prediction:** {reg_months:.1f} months

### Agent 6: Ensemble
- **Task:** Calibrated final prediction with confidence
- **Method:** Distance-calibrated blending with age-aware gating
- **Blend Weight:** {blend_weight:.2%}
- **Final:** {final_months:.0f} months | {confidence} confidence
"""

    if narrator and generate_narrative:
        agent_log += f"""
### Agent 4: Narrator (MedGemma 1.5 4B-IT)
- **Task:** Generate clinical narrative report
- **Method:** Structured evidence + X-ray image → formal radiology report
- **Anchored to:** Regressor prediction (±12m clamp)
- **Status:** {'Generated' if narrative else 'Skipped'}
"""

    return detection_img, report, narrative, agent_log, tech_json


# --- CUSTOM CSS ---
# Key fix: use CSS custom properties (var(--...)) that Gradio sets for both
# light and dark themes, and avoid hardcoding background/text colors.
custom_css = """
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Sidebar styling — uses theme variables */
.sidebar-panel {
    border-radius: 16px;
    padding: 20px;
    border: 1px solid var(--border-color-primary);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Tab styling */
.tab-nav button {
    font-size: 1.05em !important;
    font-weight: 600 !important;
    padding: 14px 24px !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid var(--color-accent) !important;
}

/* Tables — inherit from theme */
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

/* Primary button */
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

/* Markdown content — let theme handle colors */
.prose h1, .prose h2, .prose h3, .prose h4 {
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 8px;
}

/* Metrics bar at the bottom */
.metrics-bar {
    display: flex;
    justify-content: center;
    gap: 48px;
    padding: 24px 0;
    margin-top: 32px;
    border-top: 1px solid var(--border-color-primary);
    flex-wrap: wrap;
    border-radius: 12px;
    border: 1px solid var(--border-color-primary);
}

.metrics-bar .metric-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #3b82f6;
}

.metrics-bar .metric-label {
    font-size: 0.85em;
    color: var(--body-text-color-subdued);
    text-transform: uppercase;
    letter-spacing: 1px;
}
"""

# --- BUILD INTERFACE ---

# Define the theme once
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(title="Chronos-MSK | Bone Age Assessment", theme=theme, css=custom_css) as demo:

    # Header — uses explicit color on gradient background so it's always visible
    gr.HTML("""
    <div style="text-align: center; padding: 32px 0 24px 0;
                border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 32px;">
        <h1 style="margin: 0; font-size: 2.8em; font-weight: 800;
                   background: linear-gradient(135deg, #60a5fa, #3b82f6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   letter-spacing: -0.5px;">
            🦴 Chronos-MSK
        </h1>
        <p style="margin: 8px 0 0 0; color: #64748b; font-size: 1.2em; font-weight: 400;">
            Explainable Bone Age Assessment at the Edge
        </p>
        <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.95em; opacity: 0.8;">
            Powered by Google HAI-DEF &nbsp;·&nbsp; MedSigLIP-448 &nbsp;·&nbsp;
            MedGemma 1.5 4B-IT
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # Left column — Input Sidebar
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

                narrative_toggle = gr.Checkbox(
                    label="📝 Generate MedGemma Clinical Narrative",
                    value=False,
                    info="Requires LM Studio running locally",
                )

            submit_btn = gr.Button(
                "🔍 Analyze X-Ray",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"],
            )

            gr.Markdown("---")

            # Detection result
            gr.Markdown("### 🔭 Agent 1: Scout Detection")
            detection_output = gr.Image(
                label="Detected Region of Interest",
                height=320,
                interactive=False,
            )

        # Right column — Results
        with gr.Column(scale=7):
            with gr.Tabs() as tabs:

                with gr.TabItem("📊 Assessment Report", id=0):
                    report_output = gr.Markdown(
                        value="*Upload an X-ray and click Analyze to begin.*",
                    )

                with gr.TabItem("📝 MedGemma Narrative", id=1):
                    narrative_output = gr.Markdown(
                        value="*Enable narrative generation and analyze an X-ray.*",
                    )

                with gr.TabItem("🤖 Agent Log", id=2):
                    agent_log_output = gr.Markdown(
                        value="*Agent execution details will appear here after analysis.*",
                    )

                with gr.TabItem("🔧 Technical (JSON)", id=3):
                    technical_output = gr.Code(
                        value="{}",
                        language="json",
                        label="Pipeline Output",
                    )

    # Metrics bar — using Markdown for theme compatibility
    gr.Markdown("""
---

<center>

| MAE | Pearson r | Within ±12m | VRAM | Inference |
|:---:|:---:|:---:|:---:|:---:|
| **8.81m** | **0.963** | **73.2%** | **<6 GB** | **~3s** |

</center>
    """)

    # Footer
    gr.Markdown("""
---

<center>

⚠️ Research & Demonstration Only — Not a medical device

Chronos-MSK © 2025 · [GitHub](https://github.com/04RR/chronos-msk) · [Google HAI-DEF](https://developers.google.com/health-ai-developer-foundations)

</center>
    """)

    # Wire up the button
    submit_btn.click(
        fn=process_xray,
        inputs=[input_image, sex_input, narrative_toggle],
        outputs=[
            detection_output,
            report_output,
            narrative_output,
            agent_log_output,
            technical_output,
        ],
    )

# --- LAUNCH ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

# import gradio as gr
# import os
# import json
# import time
# import numpy as np
# import cv2
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from transformers import SiglipVisionModel, AutoProcessor

# from agents.agent1_scout import ScoutAgent
# from agents.agent2_radiologist import RadiologistAgent
# from agents.agent3_archivist import ArchivistAgent
# from agents.agent5_regressor import RegressorAgent
# from agents.agent6_ensemble import EnsembleAgent

# # Optional VLM
# try:
#     from agents.agent4_vlm_client import LMStudioAnthropologistAgent
#     HAS_VLM = True
# except ImportError:
#     HAS_VLM = False

# # --- CONFIG ---
# SCOUT_WEIGHTS = "weights/best_scout.pt"
# SVM_WEIGHTS = "weights/radiologist_head.pkl"
# REGRESSOR_DIR = "weights/medsiglip_sota"
# INDICES_DIR = "indices_projected_256d"
# PROJECTOR_PATH = "weights/projector_sota.pth"
# EMBED_MODEL_ID = "google/medsiglip-448"
# AVAILABLE_RACES = ["Asian", "Caucasian", "Hispanic", "Black"]
# VLM_URL = "http://10.5.0.2:1234/v1/chat/completions"
# VLM_MODEL = "medgemma-1.5-4b-it"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # --- LOAD MODELS AT STARTUP ---
# print("=" * 60)
# print("  🦴 CHRONOS-MSK — Loading Pipeline")
# print("=" * 60)

# print("  Loading Agent 1 (Scout)...")
# scout = ScoutAgent(SCOUT_WEIGHTS)

# print("  Loading Agent 2 (Radiologist)...")
# radiologist = RadiologistAgent(SVM_WEIGHTS)

# print("  Loading Agent 3 (Archivist)...")
# archivist = ArchivistAgent(INDICES_DIR, projector_path=PROJECTOR_PATH)

# print("  Loading Agent 5 (Regressor)...")
# regressor = RegressorAgent(REGRESSOR_DIR)

# print("  Loading Ensemble Agent...")
# ensemble = EnsembleAgent()

# print("  Loading MedSigLIP for retrieval embedding...")
# embed_model = SiglipVisionModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
# embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)

# narrator = None
# if HAS_VLM:
#     try:
#         import requests
#         resp = requests.get(
#             VLM_URL.replace("/chat/completions", "/models"), timeout=3
#         )
#         if resp.status_code == 200:
#             narrator = LMStudioAnthropologistAgent(api_url=VLM_URL, model_id=VLM_MODEL)
#             print("  ✅ MedGemma VLM connected")
#         else:
#             print("  ⚠️ LM Studio not responding — narrative generation disabled")
#     except Exception:
#         print("  ⚠️ LM Studio not reachable — narrative generation disabled")

# print("=" * 60)
# print("  ✅ All agents loaded. Ready to serve.")
# print("=" * 60 + "\n")


# # --- HELPER FUNCTIONS ---

# def letterbox_resize(image, size=448):
#     h, w = image.shape[:2]
#     scale = size / max(h, w)
#     nh, nw = int(h * scale), int(w * scale)
#     resized = cv2.resize(image, (nw, nh))
#     padded = np.zeros((size, size, 3), dtype=np.uint8)
#     top, left = (size - nh) // 2, (size - nw) // 2
#     padded[top : top + nh, left : left + nw] = resized
#     return padded


# def get_full_image_embedding(img_path):
#     img_bgr = cv2.imread(img_path)
#     if img_bgr is None:
#         return None
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_resized = letterbox_resize(img_rgb, 448)
#     inputs = embed_processor(images=img_resized, return_tensors="pt").to(device)
#     with torch.no_grad():
#         feat = embed_model(**inputs).pooler_output
#         feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0]


# def draw_detection(img_path):
#     """Draw scout detection bounding box on the image."""
#     img = cv2.imread(img_path)
#     if img is None:
#         return None

#     # Try all rotations like the scout does
#     best_conf = -1.0
#     best_box = None
#     best_img = None

#     rotations = [
#         None,
#         cv2.ROTATE_90_CLOCKWISE,
#         cv2.ROTATE_180,
#         cv2.ROTATE_90_COUNTERCLOCKWISE,
#     ]

#     for rot_code in rotations:
#         img_variant = (
#             cv2.rotate(img, rot_code) if rot_code is not None else img.copy()
#         )
#         results = scout.model(img_variant, verbose=False)[0]
#         for box in results.boxes:
#             if int(box.cls[0]) == 0:
#                 conf = float(box.conf[0])
#                 if conf > best_conf:
#                     best_conf = conf
#                     best_box = box.xyxy[0].cpu().numpy().astype(int)
#                     best_img = img_variant.copy()

#     if best_img is None or best_box is None:
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     x1, y1, x2, y2 = best_box
#     annotated = best_img.copy()

#     # Draw filled rectangle behind text
#     label = f"Distal Radius ({best_conf:.0%})"
#     font_scale = 0.7
#     thickness = 2
#     (tw, th), baseline = cv2.getTextSize(
#         label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
#     )

#     # Outer glow effect
#     cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 0), 4)
#     cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Label background
#     cv2.rectangle(
#         annotated,
#         (x1, y1 - th - baseline - 8),
#         (x1 + tw + 8, y1),
#         (0, 255, 0),
#         -1,
#     )
#     cv2.putText(
#         annotated,
#         label,
#         (x1 + 4, y1 - baseline - 4),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (0, 0, 0),
#         thickness,
#     )

#     # Add padding visualization
#     h, w = best_img.shape[:2]
#     pad_x = int((x2 - x1) * 0.15)
#     pad_y = int((y2 - y1) * 0.15)
#     py1 = max(0, y1 - pad_y)
#     py2 = min(h, y2 + pad_y)
#     px1 = max(0, x1 - pad_x)
#     px2 = min(w, x2 + pad_x)
#     cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 200, 0), 1)

#     return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# def format_age(months):
#     """Format age in months to a readable string."""
#     years = months / 12.0
#     if months < 24:
#         return f"{months:.0f} months"
#     else:
#         return f"{months:.0f} months ({years:.1f} years)"


# def get_confidence_color(confidence):
#     """Return color for confidence tier."""
#     colors = {
#         "HIGH": "#22c55e",
#         "MODERATE": "#eab308",
#         "LOW": "#f97316",
#         "CAUTION": "#ef4444",
#     }
#     return colors.get(confidence, "#6b7280")


# def get_confidence_emoji(confidence):
#     emojis = {
#         "HIGH": "🟢",
#         "MODERATE": "🟡",
#         "LOW": "🟠",
#         "CAUTION": "🔴",
#     }
#     return emojis.get(confidence, "⚪")


# # --- MAIN PIPELINE ---

# def process_xray(image_path, sex, generate_narrative):
#     """Run the full Chronos-MSK pipeline on a single X-ray."""

#     if image_path is None:
#         return (
#             None,
#             "## ⚠️ Please upload an X-ray image",
#             "",
#             "",
#             "{}",
#         )

#     start_time = time.time()
#     is_male = sex == "Male"

#     # ================================================================
#     # AGENT 1: SCOUT — Anatomical Detection
#     # ================================================================
#     try:
#         crop_bgr = scout.predict(image_path)
#         detection_img = draw_detection(image_path)
#         scout_status = "✅ Distal radius detected"
#     except Exception as e:
#         return (
#             None,
#             f"## ❌ Scout Detection Failed\n\n{str(e)}",
#             "",
#             "",
#             "{}",
#         )

#     # ================================================================
#     # AGENT 2: RADIOLOGIST — TW3 Staging
#     # ================================================================
#     try:
#         stage, _ = radiologist.predict(crop_bgr)
#         radio_status = f"✅ TW3 Stage: **{stage}**"
#     except Exception as e:
#         stage = "N/A"
#         radio_status = f"⚠️ Staging failed: {e}"

#     # ================================================================
#     # AGENT 3: ARCHIVIST — Visual Twin Retrieval
#     # ================================================================
#     try:
#         full_emb = get_full_image_embedding(image_path)
#         all_candidates = []
#         if full_emb is not None:
#             for race in AVAILABLE_RACES:
#                 matches = archivist.retrieve(full_emb, sex, race, top_k=3)
#                 all_candidates.extend(matches)
#         all_candidates.sort(key=lambda x: x.get("distance", 999.0))
#         best_matches = all_candidates[:5]
#         arch_status = f"✅ Retrieved {len(best_matches)} Visual Twins"
#     except Exception as e:
#         best_matches = []
#         arch_status = f"⚠️ Retrieval failed: {e}"

#     # ================================================================
#     # AGENT 5: REGRESSOR — Age Prediction
#     # ================================================================
#     try:
#         reg_months = regressor.predict(image_path, is_male)
#         reg_status = f"✅ Predicted: **{format_age(reg_months)}**"
#     except Exception as e:
#         return (
#             detection_img,
#             f"## ❌ Regression Failed\n\n{str(e)}",
#             "",
#             "",
#             "{}",
#         )

#     # ================================================================
#     # AGENT 6: ENSEMBLE — Calibrated Prediction
#     # ================================================================
#     ens_result = ensemble.predict(reg_months, best_matches)

#     final_months = ens_result["final_age_months"]
#     final_years = ens_result["final_age_years"]
#     confidence = ens_result["confidence"]
#     conf_emoji = get_confidence_emoji(confidence)
#     conf_color = get_confidence_color(confidence)
#     expected_mae = ens_result.get("expected_mae", 8.81)
#     within_12 = ens_result.get("within_12m_pct", 73.2)
#     method = ens_result.get("method", "regressor_only")
#     blend_weight = ens_result.get("blend_weight", 0)

#     elapsed = time.time() - start_time

#     # ================================================================
#     # BUILD ASSESSMENT REPORT (Markdown)
#     # ================================================================

#     # Header with result
#     report = f"""
# <div style="
#     background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
#     border-radius: 16px;
#     padding: 24px 32px;
#     margin-bottom: 20px;
#     color: white;
#     text-align: center;
# ">
#     <h1 style="margin: 0; font-size: 2.2em; font-weight: 700;">🦴 Bone Age Assessment</h1>
#     <p style="margin: 8px 0 0 0; opacity: 0.85; font-size: 1.1em;">Chronos-MSK Multi-Agent Pipeline</p>
# </div>

# <div style="
#     background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
#     border-radius: 16px;
#     padding: 28px;
#     margin-bottom: 20px;
#     border: 2px solid {conf_color};
#     text-align: center;
# ">
#     <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 2px;">Estimated Bone Age</p>
#     <h2 style="margin: 8px 0; font-size: 2.8em; color: white; font-weight: 800;">{final_months:.0f} months</h2>
#     <p style="margin: 0; color: #cbd5e1; font-size: 1.3em;">{final_years:.1f} years</p>
#     <div style="
#         display: inline-block;
#         margin-top: 12px;
#         padding: 6px 20px;
#         border-radius: 20px;
#         background: {conf_color}22;
#         border: 1px solid {conf_color};
#         color: {conf_color};
#         font-weight: 600;
#         font-size: 1em;
#     ">{conf_emoji} {confidence} Confidence</div>
# </div>
# """

#     # Accuracy context
#     report += f"""
# <div style="
#     background: #f8fafc;
#     border-radius: 12px;
#     padding: 16px 20px;
#     margin-bottom: 16px;
#     border-left: 4px solid {conf_color};
# ">
#     <p style="margin: 0; color: #334155; font-size: 0.95em;">
#         <strong>Expected Accuracy:</strong> ±{expected_mae:.0f} months &nbsp;|&nbsp;
#         <strong>{within_12:.0f}%</strong> of predictions within ±12 months &nbsp;|&nbsp;
#         <strong>Method:</strong> {method.replace('_', ' ').title()}
#         {f' (blend: {blend_weight:.0%})' if blend_weight > 0 else ''}
#     </p>
# </div>
# """

#     # Pipeline status
#     report += f"""
# ---

# ### 🔬 Pipeline Execution ({elapsed:.1f}s)

# | Agent | Model | Status |
# |---|---|---|
# | 🔭 **Scout** | YOLOv8 | {scout_status} |
# | 🩺 **Radiologist** | MedSigLIP + SVM | {radio_status} |
# | 📚 **Archivist** | MedSigLIP + FAISS | {arch_status} |
# | 🛡️ **Regressor** | MedSigLIP + LoRA | {reg_status} |
# | ⚖️ **Ensemble** | Calibrated | ✅ {confidence} confidence |

# ---

# ### 📊 Detailed Results

# | Component | Value |
# |---|---|
# | **Final Estimate** | **{final_months:.0f} months ({final_years:.1f} years)** |
# | Regressor Prediction | {reg_months:.1f} months ({reg_months/12:.1f} years) |
# | Patient Sex | {sex} |
# | TW3 Maturity Stage | {stage} |
# | Confidence Tier | {conf_emoji} {confidence} |
# | Expected MAE | ±{expected_mae:.0f} months |
# | Processing Time | {elapsed:.1f} seconds |
# """

#     # Visual Twins section
#     if best_matches:
#         avg_dist = np.mean([m.get("distance", 999) for m in best_matches])
#         valid_ages = [m.get("age_months", -1) for m in best_matches if m.get("age_months", -1) > 0]
#         arch_pred = np.mean(valid_ages) if valid_ages else None
#         agreement = abs(reg_months - arch_pred) if arch_pred else None

#         report += f"""
# ---

# ### 🔍 Visual Twin Analysis (Atlas Retrieval)

# """
#         if arch_pred:
#             report += f"**Atlas Peer Average:** {arch_pred:.0f} months ({arch_pred/12:.1f} years)"
#             if agreement is not None:
#                 if agreement < 12:
#                     report += f" — ✅ Agrees with regressor (Δ = {agreement:.0f}m)\n\n"
#                 elif agreement < 24:
#                     report += f" — ⚠️ Moderate disagreement (Δ = {agreement:.0f}m)\n\n"
#                 else:
#                     report += f" — 🔴 Significant disagreement (Δ = {agreement:.0f}m)\n\n"

#         report += "| Rank | Age | Distance | Demographic | Quality |\n"
#         report += "|---|---|---|---|---|\n"

#         for i, m in enumerate(best_matches):
#             age_m = m.get("age_months", -1)
#             dist = m.get("distance", -1)
#             partition = m.get("partition", "?")

#             if dist < 0.10:
#                 quality = "🟢 Excellent"
#             elif dist < 0.15:
#                 quality = "🟡 Good"
#             elif dist < 0.25:
#                 quality = "🟠 Fair"
#             else:
#                 quality = "🔴 Poor"

#             report += (
#                 f"| #{i+1} | {age_m:.0f}m ({age_m/12:.1f}y) | "
#                 f"{dist:.4f} | {partition} | {quality} |\n"
#             )

#         report += f"\n**Average Retrieval Distance:** {avg_dist:.4f}\n"
#     else:
#         report += "\n---\n\n*No atlas matches available for this query.*\n"

#     # Explanation
#     explanation = ens_result.get("explanation", "")
#     if explanation:
#         report += f"""
# ---

# ### 💡 Clinical Context

# {explanation}
# """

#     # Disclaimer
#     report += """
# ---

# <div style="
#     background: #fef3c7;
#     border-radius: 8px;
#     padding: 12px 16px;
#     margin-top: 16px;
#     border-left: 4px solid #f59e0b;
# ">
#     <p style="margin: 0; color: #92400e; font-size: 0.85em;">
#         ⚠️ <strong>Research Tool Only</strong> — This is not a medical device.
#         Results require clinical interpretation by a qualified professional.
#         Powered by Google HAI-DEF (MedSigLIP-448 + MedGemma 1.5 4B-IT).
#     </p>
# </div>
# """

#     # ================================================================
#     # VLM NARRATIVE (Optional)
#     # ================================================================
#     narrative = ""
#     if generate_narrative:
#         if narrator:
#             try:
#                 vlm_report = narrator.analyze(
#                     image_path=image_path,
#                     sex=sex,
#                     race="Unknown",
#                     stage=stage,
#                     matches=best_matches,
#                     reg_age_months=reg_months,
#                 )
#                 vlm_text = vlm_report.get("visual_analysis", "")
#                 vlm_reasoning = vlm_report.get("adjustment_reasoning", "")
#                 vlm_flag = vlm_report.get("flag", "")
#                 vlm_age = vlm_report.get("final_age_months", reg_months)
#                 vlm_unclamped = vlm_report.get("unclamped_vlm_age", vlm_age)

#                 narrative = f"""## 📝 MedGemma Clinical Narrative

# ### Visual Analysis
# {vlm_text}

# ### Clinical Reasoning
# {vlm_reasoning}

# ### VLM Assessment
# - **VLM Estimate:** {vlm_age:.0f} months ({vlm_age/12:.1f} years)
# """
#                 if abs(vlm_unclamped - vlm_age) > 0.1:
#                     narrative += f"- **Pre-clamp Estimate:** {vlm_unclamped:.0f} months (clamped to ±12m of regressor)\n"

#                 narrative += f"- **Status:** {vlm_flag}\n"
#                 narrative += f"\n*Generated by MedGemma 1.5 4B-IT via LM Studio*\n"

#             except Exception as e:
#                 narrative = f"## ⚠️ Narrative Generation Failed\n\n{str(e)}"
#         else:
#             narrative = (
#                 "## ℹ️ MedGemma Not Available\n\n"
#                 "To enable clinical narrative generation:\n\n"
#                 "1. Install [LM Studio](https://lmstudio.ai/)\n"
#                 "2. Load the `medgemma-1.5-4b-it` model\n"
#                 "3. Start the local server\n"
#                 "4. Restart this application\n"
#             )

#     # ================================================================
#     # TECHNICAL DETAILS (JSON)
#     # ================================================================
#     tech_details = {
#         "pipeline": {
#             "final_age_months": final_months,
#             "final_age_years": final_years,
#             "confidence": confidence,
#             "method": method,
#             "processing_time_seconds": round(elapsed, 2),
#         },
#         "regressor": {
#             "prediction_months": reg_months,
#             "model": "MedSigLIP-448 + LoRA (DLDL, 228 bins)",
#         },
#         "radiologist": {
#             "tw3_stage": stage,
#             "model": "MedSigLIP-448 + SVM",
#         },
#         "retrieval": {
#             "n_matches": len(best_matches),
#             "avg_distance": round(float(np.mean([m.get("distance", 999) for m in best_matches])), 4) if best_matches else None,
#             "match_ages": [m.get("age_months", -1) for m in best_matches],
#             "match_distances": [round(m.get("distance", -1), 4) for m in best_matches],
#             "match_partitions": [m.get("partition", "?") for m in best_matches],
#         },
#         "ensemble": {
#             k: v for k, v in ens_result.items()
#             if k not in ["gt_months", "ae_ensemble", "ae_regressor", "ae_archivist"]
#         },
#         "system": {
#             "device": device,
#             "models_loaded": [
#                 "MedSigLIP-448 (feature extraction + regression + retrieval)",
#                 "YOLOv8 (scout detection)",
#                 "SVM (TW3 staging)",
#                 "SOTA Projector (1152→256D)",
#             ],
#             "vlm_available": narrator is not None,
#         },
#     }

#     tech_json = json.dumps(tech_details, indent=2, default=str)

#     # Agent log
#     agent_log = f"""## 🤖 Agent Execution Log

# **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
# **Total Processing Time:** {elapsed:.2f} seconds
# **Device:** {device}

# ---

# ### Agent 1: Scout (YOLOv8)
# - **Task:** Detect distal radius region in X-ray
# - **Method:** Rotation-invariant search (0°, 90°, 180°, 270°)
# - **Padding:** 15% bounding box expansion
# - **Result:** {scout_status}

# ### Agent 2: Radiologist (MedSigLIP + SVM)
# - **Task:** TW3 skeletal maturity staging
# - **Method:** Frozen MedSigLIP embeddings (1152-D) → SVM classifier
# - **TTA:** Original + horizontal flip, averaged probabilities
# - **Result:** Stage {stage}

# ### Agent 3: Archivist (MedSigLIP + FAISS)
# - **Task:** Retrieve demographically-matched "Visual Twins"
# - **Method:** Full-image embedding → 256-D SOTA projection → L2 search
# - **Partitions Searched:** {', '.join(f'{sex}_{r}' for r in AVAILABLE_RACES)}
# - **Matches Found:** {len(best_matches)}
# - **Avg Distance:** {f"{np.mean([m.get('distance', 999) for m in best_matches]):.4f}" if best_matches else "N/A"}

# ### Agent 5: Regressor (MedSigLIP + LoRA)
# - **Task:** Predict bone age as probability distribution
# - **Method:** DoRA-adapted MedSigLIP with sex conditioning
# - **Output:** Softmax over 228 bins → expected value
# - **TTA:** Original + horizontal flip, averaged
# - **Prediction:** {reg_months:.1f} months

# ### Agent 6: Ensemble
# - **Task:** Calibrated final prediction with confidence
# - **Method:** Distance-calibrated blending with age-aware gating
# - **Blend Weight:** {blend_weight:.2%}
# - **Final:** {final_months:.0f} months | {confidence} confidence
# """

#     if narrator and generate_narrative:
#         agent_log += f"""
# ### Agent 4: Narrator (MedGemma 1.5 4B-IT)
# - **Task:** Generate clinical narrative report
# - **Method:** Structured evidence + X-ray image → formal radiology report
# - **Anchored to:** Regressor prediction (±12m clamp)
# - **Status:** {'Generated' if narrative else 'Skipped'}
# """

#     return detection_img, report, narrative, agent_log, tech_json


# # --- CUSTOM CSS ---
# custom_css = """
# /* Global */
# .gradio-container {
#     max-width: 1400px !important;
#     font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
# }

# /* Header */
# .header-container {
#     text-align: center;
#     padding: 20px 0;
# }

# /* Tabs */
# .tab-nav button {
#     font-size: 1em !important;
#     font-weight: 600 !important;
#     padding: 12px 24px !important;
# }

# .tab-nav button.selected {
#     border-bottom: 3px solid #2563eb !important;
#     color: #2563eb !important;
# }

# /* Cards */
# .card {
#     border-radius: 12px;
#     border: 1px solid #e2e8f0;
#     padding: 16px;
# }

# /* Make markdown tables look better */
# table {
#     border-collapse: collapse;
#     width: 100%;
#     margin: 12px 0;
# }

# th {
#     background: #f1f5f9;
#     padding: 10px 14px;
#     text-align: left;
#     font-weight: 600;
#     color: #334155;
#     border-bottom: 2px solid #cbd5e1;
# }

# td {
#     padding: 8px 14px;
#     border-bottom: 1px solid #e2e8f0;
#     color: #475569;
# }

# tr:hover td {
#     background: #f8fafc;
# }

# /* Buttons */
# .primary-btn {
#     background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
#     border: none !important;
#     font-size: 1.1em !important;
#     font-weight: 600 !important;
#     padding: 14px 32px !important;
#     border-radius: 12px !important;
#     transition: all 0.2s !important;
# }

# .primary-btn:hover {
#     transform: translateY(-1px) !important;
#     box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
# }

# /* JSON viewer */
# .json-viewer pre {
#     font-size: 0.85em !important;
#     line-height: 1.5 !important;
# }

# /* Footer */
# .footer {
#     text-align: center;
#     padding: 16px;
#     color: #94a3b8;
#     font-size: 0.85em;
# }
# """


# # --- BUILD INTERFACE ---

# with gr.Blocks(
#     title="Chronos-MSK | Bone Age Assessment",
#     theme=gr.themes.Soft(
#         primary_hue="blue",
#         secondary_hue="slate",
#         neutral_hue="slate",
#         font=gr.themes.GoogleFont("Inter"),
#     ),
#     css=custom_css,
# ) as demo:

#     # Header
#     gr.HTML("""
#     <div style="
#         text-align: center;
#         padding: 24px 0 16px 0;
#         border-bottom: 1px solid #e2e8f0;
#         margin-bottom: 20px;
#     ">
#         <h1 style="
#             margin: 0;
#             font-size: 2.4em;
#             font-weight: 800;
#             background: linear-gradient(135deg, #1e3a5f, #3b82f6);
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#             letter-spacing: -0.5px;
#         ">🦴 Chronos-MSK</h1>
#         <p style="
#             margin: 6px 0 0 0;
#             color: #64748b;
#             font-size: 1.15em;
#             font-weight: 400;
#         ">Explainable Bone Age Assessment at the Edge</p>
#         <p style="
#             margin: 4px 0 0 0;
#             color: #94a3b8;
#             font-size: 0.9em;
#         ">Powered by Google HAI-DEF &nbsp;·&nbsp; MedSigLIP-448 &nbsp;·&nbsp; MedGemma 1.5 4B-IT</p>
#     </div>
#     """)

#     with gr.Row(equal_height=False):

#         # Left column — Input
#         with gr.Column(scale=2):
#             gr.Markdown("### 📤 Input")

#             input_image = gr.Image(
#                 type="filepath",
#                 label="Hand/Wrist X-Ray",
#                 height=380,
#                 sources=["upload", "clipboard"],
#             )

#             with gr.Row():
#                 sex_input = gr.Radio(
#                     choices=["Male", "Female"],
#                     label="Biological Sex",
#                     value="Male",
#                     interactive=True,
#                 )

#             narrative_toggle = gr.Checkbox(
#                 label="📝 Generate MedGemma Clinical Narrative",
#                 value=False,
#                 info="Requires LM Studio running with medgemma-1.5-4b-it loaded",
#             )

#             submit_btn = gr.Button(
#                 "🔍 Analyze X-Ray",
#                 variant="primary",
#                 size="lg",
#                 elem_classes=["primary-btn"],
#             )

#             # Detection result
#             gr.Markdown("### 🔭 Agent 1: Scout Detection")
#             detection_output = gr.Image(
#                 label="Detected Region of Interest",
#                 height=380,
#                 interactive=False,
#             )

#         # Right column — Results
#         with gr.Column(scale=3):
#             with gr.Tabs() as tabs:

#                 with gr.TabItem("📊 Assessment Report", id=0):
#                     report_output = gr.Markdown(
#                         value="*Upload an X-ray and click Analyze to begin.*",
#                     )

#                 with gr.TabItem("📝 MedGemma Narrative", id=1):
#                     narrative_output = gr.Markdown(
#                         value="*Enable narrative generation and analyze an X-ray.*",
#                     )

#                 with gr.TabItem("🤖 Agent Log", id=2):
#                     agent_log_output = gr.Markdown(
#                         value="*Agent execution details will appear here after analysis.*",
#                     )

#                 with gr.TabItem("🔧 Technical (JSON)", id=3):
#                     technical_output = gr.Code(
#                         value="{}",
#                         language="json",
#                         label="Pipeline Output",
#                     )

#     # Metrics bar
#     gr.HTML("""
#     <div style="
#         display: flex;
#         justify-content: center;
#         gap: 32px;
#         padding: 16px 0;
#         margin-top: 16px;
#         border-top: 1px solid #e2e8f0;
#         flex-wrap: wrap;
#     ">
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">8.81m</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">MAE</div>
#         </div>
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">0.963</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">Pearson r</div>
#         </div>
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">73.2%</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">Within ±12m</div>
#         </div>
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">&lt;6 GB</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">VRAM</div>
#         </div>
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">~3s</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">Inference</div>
#         </div>
#         <div style="text-align: center;">
#             <div style="font-size: 1.4em; font-weight: 700; color: #1e3a5f;">Offline</div>
#             <div style="font-size: 0.8em; color: #94a3b8;">No Internet</div>
#         </div>
#     </div>
#     """)

#     # Footer
#     gr.HTML("""
#     <div style="
#         text-align: center;
#         padding: 12px;
#         color: #94a3b8;
#         font-size: 0.8em;
#         border-top: 1px solid #f1f5f9;
#         margin-top: 8px;
#     ">
#         ⚠️ Research & Demonstration Only — Not a medical device<br>
#         Chronos-MSK © 2025 &nbsp;·&nbsp;
#         <a href="https://github.com/04RR/chronos-msk" style="color: #64748b;">GitHub</a> &nbsp;·&nbsp;
#         <a href="https://developers.google.com/health-ai-developer-foundations" style="color: #64748b;">Google HAI-DEF</a>
#     </div>
#     """)

#     # Wire up the button
#     submit_btn.click(
#         fn=process_xray,
#         inputs=[input_image, sex_input, narrative_toggle],
#         outputs=[
#             detection_output,
#             report_output,
#             narrative_output,
#             agent_log_output,
#             technical_output,
#         ],
#     )


# # --- LAUNCH ---
# if __name__ == "__main__":
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=True,
#         show_error=True,
#     )
