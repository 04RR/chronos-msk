"""
evaluate.py
===========
Consolidated evaluation for Chronos-MSK.
Supports three modes:
  1. Fast (ensemble only, no VLM)
  2. Full (ensemble + VLM narratives)
  3. Metrics (compute from existing results)

Usage:
  python evaluate.py --mode fast
  python evaluate.py --mode full --vlm-url http://10.5.0.2:1234/v1/chat/completions
  python evaluate.py --mode metrics --results evaluation_results/metrics.csv
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from scipy import stats
from transformers import SiglipVisionModel, AutoProcessor

from agents.agent1_scout import ScoutAgent
from agents.agent2_radiologist import RadiologistAgent
from agents.agent3_archivist import ArchivistAgent
from agents.agent5_regressor import RegressorAgent
from agents.agent6_ensemble import EnsembleAgent

# --- CONFIGURATION ---
VAL_CSV = "data/boneage_val.csv"
IMAGE_DIR = "data/RSNA_val/images"
OUTPUT_DIR = "evaluation_results"

INDICES_DIR = "indices_projected_256d"
PROJECTOR_PATH = "weights/projector_sota.pth"
SCOUT_WEIGHTS = "weights/best_scout.pt"
SVM_WEIGHTS = "weights/radiologist_head.pkl"
REGRESSOR_DIR = "weights/medsiglip_sota"
EMBED_MODEL_ID = "google/medsiglip-448"

AVAILABLE_RACES = ["Asian", "Caucasian", "Hispanic", "Black"]


def letterbox_resize(image, size=448):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded[top : top + nh, left : left + nw] = resized
    return padded


def get_full_image_embedding(img_path, embed_model, processor, device):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = letterbox_resize(img_rgb, 448)
    inputs = processor(images=img_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = embed_model(**inputs).pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


# ================================================================
# PHASE 1: LOCAL INFERENCE
# ================================================================
def run_phase1(cache_file, match_file):
    """Run all local agents, cache results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading agents...")
    scout = ScoutAgent(SCOUT_WEIGHTS)
    radiologist = RadiologistAgent(SVM_WEIGHTS)
    regressor = RegressorAgent(REGRESSOR_DIR)
    archivist = ArchivistAgent(INDICES_DIR, projector_path=PROJECTOR_PATH)

    print(f"Loading MedSigLIP for full-image embedding...")
    embed_model = SiglipVisionModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
    embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)

    df = pd.read_csv(VAL_CSV)
    cache_data = []
    match_cache = {}

    print(f"Processing {len(df)} images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        case_id = str(row["ID"])
        img_path = os.path.join(IMAGE_DIR, f"{case_id}.png")
        if not os.path.exists(img_path):
            continue

        sex_str = "Male" if row["Male"] else "Female"
        try:
            crop_bgr = scout.predict(img_path)
            stage, _ = radiologist.predict(crop_bgr)

            full_embedding = get_full_image_embedding(
                img_path, embed_model, embed_processor, device
            )
            if full_embedding is None:
                continue

            all_candidates = []
            for race in AVAILABLE_RACES:
                race_matches = archivist.retrieve(full_embedding, sex_str, race, top_k=3)
                all_candidates.extend(race_matches)
            all_candidates.sort(key=lambda x: x.get("distance", 999.0))
            best_matches = all_candidates[:5]

            reg_age_months = regressor.predict(img_path, row["Male"])

            cache_data.append({
                "case_id": case_id,
                "img_path": img_path,
                "sex_str": sex_str,
                "stage": stage,
                "reg_age_months": reg_age_months,
                "gt_months": row["Boneage"],
            })
            match_cache[case_id] = best_matches

        except Exception as e:
            print(f"  Error on {case_id}: {e}")

    del embed_model
    torch.cuda.empty_cache()

    cache_df = pd.DataFrame(cache_data)
    cache_df.to_csv(cache_file, index=False)
    with open(match_file, "w") as f:
        json.dump(match_cache, f)

    print(f"Phase 1 complete: {len(cache_df)} cases cached")
    return cache_df, match_cache


# ================================================================
# PHASE 2: ENSEMBLE SCORING
# ================================================================
def run_ensemble(cache_df, match_cache):
    """Score all cases with the ensemble agent."""
    ensemble = EnsembleAgent()
    results = []

    for _, row in tqdm(cache_df.iterrows(), total=len(cache_df), desc="Scoring"):
        case_id = str(row["case_id"])
        matches = match_cache.get(case_id, [])

        result = ensemble.predict(
            reg_age_months=float(row["reg_age_months"]),
            matches=matches,
            gt_months=float(row["gt_months"]),
        )
        result["ID"] = case_id
        result["sex_str"] = row.get("sex_str", "Unknown")
        results.append(result)

    return pd.DataFrame(results)


# ================================================================
# PHASE 3: VLM INFERENCE (optional)
# ================================================================
def run_vlm(cache_df, match_cache, res_df, vlm_url, vlm_model):
    """Add VLM narratives to existing results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from agents.agent4_vlm_client import LMStudioAnthropologistAgent

    vlm_cache_dir = os.path.join(OUTPUT_DIR, "vlm_responses")
    os.makedirs(vlm_cache_dir, exist_ok=True)

    anthropologist = LMStudioAnthropologistAgent(api_url=vlm_url, model_id=vlm_model)

    vlm_results = []

    def process_case(row_data):
        case_id = str(row_data["case_id"])
        json_path = os.path.join(vlm_cache_dir, f"{case_id}.json")

        matches = match_cache.get(case_id, [])
        reg_age = float(row_data["reg_age_months"])

        # Check cache
        vlm_report = None
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    vlm_report = json.load(f)
                if "final_age_months" not in vlm_report:
                    vlm_report = None
            except:
                vlm_report = None

        if vlm_report is None:
            vlm_report = anthropologist.analyze(
                image_path=row_data["img_path"],
                sex=row_data["sex_str"],
                race="Unknown (RSNA)",
                stage=None,
                matches=matches,
                reg_age_months=reg_age,
            )
            with open(json_path, "w") as f:
                json.dump(vlm_report, f, indent=2)

        try:
            vlm_age = float(vlm_report.get("final_age_months", reg_age))
        except:
            vlm_age = reg_age

        return {
            "ID": case_id,
            "VLM_Clamped_Months": vlm_age,
            "VLM_Unclamped_Months": float(vlm_report.get("unclamped_vlm_age", vlm_age)),
            "VLM_Flag": vlm_report.get("flag", "N/A"),
            "VLM_Analysis": str(vlm_report.get("visual_analysis", ""))[:200],
            "VLM_Reasoning": str(vlm_report.get("adjustment_reasoning", ""))[:200],
        }

    print(f"Running VLM inference for {len(cache_df)} cases...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_case, row): row["case_id"]
            for _, row in cache_df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(cache_df)):
            try:
                vlm_results.append(future.result())
            except:
                pass

    vlm_df = pd.DataFrame(vlm_results)
    res_df["ID"] = res_df["ID"].astype(str)
    vlm_df["ID"] = vlm_df["ID"].astype(str)
    merged = res_df.merge(vlm_df, on="ID", how="left")

    # Compute VLM errors
    if "VLM_Clamped_Months" in merged.columns:
        merged["AE_VLM_Clamped"] = abs(
            merged["VLM_Clamped_Months"] - merged["gt_months"]
        )
    if "VLM_Unclamped_Months" in merged.columns:
        merged["AE_VLM_Unclamped"] = abs(
            merged["VLM_Unclamped_Months"] - merged["gt_months"]
        )

    return merged


# ================================================================
# REPORT
# ================================================================
def print_report(res_df, include_vlm=False):
    """Print comprehensive evaluation report."""
    gt = res_df["gt_months"].values
    pred = res_df["final_age_months"].values
    ae = res_df["ae_ensemble"].values
    reg_ae = res_df["ae_regressor"].values
    n = len(res_df)

    r, _ = stats.pearsonr(gt, pred)
    arch_ae = res_df["ae_archivist"].dropna().values

    print("\n" + "#" * 70)
    print("  CHRONOS-MSK EVALUATION REPORT")
    print("#" * 70)

    # Core metrics
    print(f"\n  CORE ACCURACY (n={n})")
    print(f"  {'─'*55}")
    print(f"  {'Metric':<30s} {'Value':>20s}")
    print(f"  {'─'*55}")
    print(f"  {'MAE':<30s} {np.mean(ae):>15.2f} months")
    print(f"  {'Regressor MAE':<30s} {np.mean(reg_ae):>15.2f} months")
    print(f"  {'Archivist MAE':<30s} {np.mean(arch_ae):>15.2f} months")
    print(f"  {'Median AE':<30s} {np.median(ae):>15.2f} months")
    print(f"  {'RMSE':<30s} {np.sqrt(np.mean(ae**2)):>15.2f} months")
    print(f"  {'Pearson r':<30s} {r:>15.4f}")
    print(f"  {'R squared':<30s} {r**2:>15.4f}")

    # Threshold accuracy
    print(f"\n  THRESHOLD ACCURACY")
    print(f"  {'─'*55}")
    for t in [6, 12, 18, 24]:
        ens_pct = 100 * (ae <= t).sum() / n
        reg_pct = 100 * (reg_ae <= t).sum() / n
        print(f"  Within +/-{t:2d}mo:  Ensemble={ens_pct:.1f}%  Regressor={reg_pct:.1f}%")

    # Confidence calibration
    print(f"\n  CONFIDENCE CALIBRATION")
    print(f"  {'─'*55}")
    for conf in ["HIGH", "MODERATE", "LOW", "CAUTION"]:
        subset = res_df[res_df["confidence"] == conf]
        if len(subset) > 0:
            conf_mae = subset["ae_ensemble"].mean()
            w12 = 100 * (subset["ae_ensemble"] <= 12).sum() / len(subset)
            print(f"  {conf:<12s} (n={len(subset):4d}): MAE={conf_mae:.2f}m, "
                  f"Within+/-12m={w12:.1f}%")

    # Age-stratified
    print(f"\n  AGE-STRATIFIED MAE")
    print(f"  {'─'*55}")
    for lo, hi, label in [(0, 60, "0-5y"), (60, 120, "5-10y"),
                           (120, 180, "10-15y"), (180, 228, "15-19y")]:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() > 0:
            e_mae = np.mean(ae[mask])
            r_mae = np.mean(reg_ae[mask])
            print(f"  {label:8s} (n={mask.sum():4d}): "
                  f"Ensemble={e_mae:.2f}m, Regressor={r_mae:.2f}m")

    # Sex-stratified
    print(f"\n  SEX-STRATIFIED MAE")
    print(f"  {'─'*55}")
    for sex in ["Male", "Female"]:
        mask = res_df["sex_str"] == sex
        if mask.sum() > 0:
            print(f"  {sex:<10s} (n={mask.sum():4d}): "
                  f"Ensemble={np.mean(ae[mask]):.2f}m, "
                  f"Regressor={np.mean(reg_ae[mask]):.2f}m")

    # VLM metrics (if available)
    if include_vlm and "AE_VLM_Clamped" in res_df.columns:
        vlm_ae = res_df["AE_VLM_Clamped"].dropna().values
        if len(vlm_ae) > 0:
            print(f"\n  VLM METRICS")
            print(f"  {'─'*55}")
            print(f"  VLM Clamped MAE:    {np.mean(vlm_ae):.2f}m")
            improved = (vlm_ae < reg_ae[:len(vlm_ae)]).sum()
            degraded = (vlm_ae > reg_ae[:len(vlm_ae)]).sum()
            print(f"  VLM improved:       {improved}/{len(vlm_ae)}")
            print(f"  VLM degraded:       {degraded}/{len(vlm_ae)}")

    # Distance analysis
    if "avg_retrieval_distance" in res_df.columns:
        dists = res_df["avg_retrieval_distance"].dropna()
        arch_errors = res_df["ae_archivist"].dropna()
        common = dists.index.intersection(arch_errors.index)
        if len(common) > 10:
            d_corr, _ = stats.pearsonr(dists.loc[common], arch_errors.loc[common])
            print(f"\n  Distance-Error correlation: r={d_corr:+.4f}")

    # Summary
    print(f"\n{'#'*70}")
    within12 = 100 * (ae <= 12).sum() / n
    print(f"  MAE: {np.mean(ae):.2f}m | Pearson r: {r:.4f} | Within+/-12m: {within12:.1f}%")
    print(f"{'#'*70}\n")

    # Export markdown
    markdown = f"""| Metric | Value |
|---|---|
| **MAE** | **{np.mean(ae):.2f} months ({np.mean(ae)/12:.2f} years)** |
| Median AE | {np.median(ae):.2f} months |
| RMSE | {np.sqrt(np.mean(ae**2)):.2f} months |
| Pearson r | {r:.4f} |
| R squared | {r**2:.4f} |
| Within +/-6m | {100*(ae<=6).sum()/n:.1f}% |
| Within +/-12m | {100*(ae<=12).sum()/n:.1f}% |
| Within +/-24m | {100*(ae<=24).sum()/n:.1f}% |
"""
    md_path = os.path.join(OUTPUT_DIR, "metrics.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Metrics saved to {md_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Chronos-MSK Evaluation")
    parser.add_argument(
        "--mode", choices=["fast", "full", "metrics"], default="fast",
        help="fast=ensemble only, full=ensemble+VLM, metrics=from existing CSV"
    )
    parser.add_argument("--vlm-url", default="http://localhost:1234/v1/chat/completions")
    parser.add_argument("--vlm-model", default="medgemma-1.5-4b-it")
    parser.add_argument("--results", default=None, help="Path to existing results CSV")
    parser.add_argument("--val-csv", default=VAL_CSV)
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    args = parser.parse_args()

    global VAL_CSV, IMAGE_DIR
    VAL_CSV = args.val_csv
    IMAGE_DIR = args.image_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cache_file = os.path.join(OUTPUT_DIR, "phase1_cache.csv")
    match_file = os.path.join(OUTPUT_DIR, "phase1_matches.json")

    if args.mode == "metrics":
        if args.results and os.path.exists(args.results):
            res_df = pd.read_csv(args.results)
            print_report(res_df, include_vlm="AE_VLM_Clamped" in res_df.columns)
        else:
            print("Provide --results path to an existing CSV")
        return

    # Phase 1: Local inference
    if os.path.exists(cache_file) and os.path.exists(match_file):
        print(f"Loading cached Phase 1 results...")
        cache_df = pd.read_csv(cache_file)
        with open(match_file) as f:
            match_cache = json.load(f)
    else:
        cache_df, match_cache = run_phase1(cache_file, match_file)

    # Phase 2: Ensemble
    print(f"\nScoring {len(cache_df)} cases...")
    res_df = run_ensemble(cache_df, match_cache)

    # Phase 3: VLM (if full mode)
    if args.mode == "full":
        res_df = run_vlm(cache_df, match_cache, res_df, args.vlm_url, args.vlm_model)

    # Save and report
    output_csv = os.path.join(OUTPUT_DIR, "results.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    print_report(res_df, include_vlm=(args.mode == "full"))


if __name__ == "__main__":
    main()
