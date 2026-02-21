"""
vlm_eval.py
===========
Full pipeline evaluation with VLM narrative reports.
VLM generates clinical narratives, NOT numeric predictions.
Numeric prediction comes from the regressor (clamped as safety net).
"""

import os
import json
import pandas as pd
import numpy as np
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from transformers import SiglipVisionModel, AutoProcessor

from agents.agent1_scout import ScoutAgent
from agents.agent2_radiologist import RadiologistAgent
from agents.agent3_archivist import ArchivistAgent
from agents.agent5_regressor import RegressorAgent
from agents.agent4_vlm_client import LMStudioAnthropologistAgent
from agents.agent6_ensemble import EnsembleAgent

# --- CONFIGURATION ---
VAL_CSV = "data/boneage_val.csv"
IMAGE_DIR = "/mnt/e/chronos-msk/data/RSNA Bone Age+Anatomical ROIS/RSNA_val/images"
OUTPUT_DIR = "evaluation_results_vlm_final"
VLM_CACHE_DIR = os.path.join(OUTPUT_DIR, "vlm_responses")
CACHE_FILE = os.path.join(OUTPUT_DIR, "phase1_cache.csv")
MATCH_CACHE_FILE = os.path.join(OUTPUT_DIR, "phase1_matches.json")

INDICES_DIR = "indices_projected_256d"
PROJECTOR_PATH = "weights/projector_sota.pth"
SCOUT_WEIGHTS = "weights/best_scout.pt"
SVM_WEIGHTS = "weights/radiologist_head.pkl"
REGRESSOR_DIR = "weights/medsiglip_sota"
EMBED_MODEL_ID = "google/medsiglip-448"

LM_STUDIO_URL = "http://10.5.0.2:1234/v1/chat/completions"
LM_MODEL_ID = "medgemma-1.5-4b-it"
MAX_WORKERS = 4
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


def load_matches(case_id, match_cache):
    return match_cache.get(str(case_id), [])


def process_vlm_request(anthropologist, ensemble, row_data, match_cache):
    """
    Process a single case:
    1. Ensemble produces the NUMERIC prediction
    2. VLM produces the NARRATIVE report explaining it
    """
    case_id = str(row_data["case_id"])
    json_path = os.path.join(VLM_CACHE_DIR, f"{case_id}.json")

    matches = load_matches(case_id, match_cache)
    reg_age = float(row_data["reg_age_months"])
    gt_months = float(row_data["gt_months"])

    # Step 1: Ensemble prediction (deterministic, no VLM)
    ens_result = ensemble.predict(
        reg_age_months=reg_age,
        matches=matches,
        gt_months=gt_months,
    )

    final_age = ens_result["final_age_months"]

    # Step 2: VLM narrative (cached)
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
            stage=row_data.get("stage", None),
            matches=matches,
            reg_age_months=reg_age,
        )
        with open(json_path, "w") as f:
            json.dump(vlm_report, f, indent=2)

    # Step 3: Extract VLM's opinion (for analysis only)
    try:
        vlm_age = float(vlm_report.get("final_age_months", reg_age))
    except (TypeError, ValueError):
        vlm_age = reg_age

    vlm_unclamped = float(vlm_report.get("unclamped_vlm_age", vlm_age))
    vlm_flag = vlm_report.get("flag", "N/A")
    vlm_analysis = vlm_report.get("visual_analysis", "")
    vlm_conflict = vlm_report.get("conflict_resolution", "")
    vlm_reasoning = vlm_report.get("adjustment_reasoning", "")

    # Archivist metric
    valid_ages = [m.get("age_months", -1) for m in matches if m.get("age_months", -1) > 0]
    archivist_mae = np.nan
    archivist_pred = np.nan
    if valid_ages:
        archivist_pred = np.mean(valid_ages)
        archivist_mae = abs(archivist_pred - gt_months)

    return {
        # IDs
        "ID": case_id,
        # Ground truth
        "GT_Months": gt_months,
        # Predictions
        "Ensemble_Months": final_age,
        "Reg_Months": reg_age,
        "Archivist_Months": archivist_pred,
        "VLM_Clamped_Months": vlm_age,
        "VLM_Unclamped_Months": vlm_unclamped,
        # Errors
        "AE_Ensemble": abs(final_age - gt_months),
        "AE_Regressor": abs(reg_age - gt_months),
        "AE_Archivist": archivist_mae,
        "AE_VLM_Clamped": abs(vlm_age - gt_months),
        "AE_VLM_Unclamped": abs(vlm_unclamped - gt_months),
        # Metadata
        "Confidence": ens_result.get("confidence", "N/A"),
        "Method": ens_result.get("method", "N/A"),
        "Blend_Weight": ens_result.get("blend_weight", 0),
        "Agreement_Months": ens_result.get("agreement_months", np.nan),
        "Avg_Distance": ens_result.get("avg_retrieval_distance", np.nan),
        # VLM narrative
        "VLM_Flag": vlm_flag,
        "VLM_Analysis": str(vlm_analysis)[:200],
        "VLM_Reasoning": str(vlm_reasoning)[:200],
        "VLM_Conflict": str(vlm_conflict)[:200],
        # Match details
        "Match_Ages": str([m.get("age_months", -1) for m in matches]),
        "Match_Distances": str([round(m.get("distance", -1), 4) for m in matches]),
        "N_Matches": len(matches),
        "Success": True,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VLM_CACHE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # PHASE 1: LOCAL INFERENCE (reuse from embedding_fix if available)
    # ================================================================
    match_cache = {}

    # Try to reuse embedding_fix cache first
    emb_cache = "evaluation_results_embedding_fix/cache.csv"
    emb_match = "evaluation_results_embedding_fix/matches.json"

    if os.path.exists(CACHE_FILE) and os.path.exists(MATCH_CACHE_FILE):
        print(f"⚡ Using existing VLM Phase 1 cache...")
        cache_df = pd.read_csv(CACHE_FILE)
        with open(MATCH_CACHE_FILE, "r") as f:
            match_cache = json.load(f)
    elif os.path.exists(emb_cache) and os.path.exists(emb_match):
        print(f"⚡ Reusing embedding_fix Phase 1 cache...")
        cache_df = pd.read_csv(emb_cache)
        with open(emb_match, "r") as f:
            match_cache = json.load(f)
        # Save a copy for this output dir
        cache_df.to_csv(CACHE_FILE, index=False)
        with open(MATCH_CACHE_FILE, "w") as f:
            json.dump(match_cache, f)
    else:
        print("🐢 Running Phase 1 (local inference)...")

        scout = ScoutAgent(SCOUT_WEIGHTS)
        radiologist = RadiologistAgent(SVM_WEIGHTS)
        regressor = RegressorAgent(REGRESSOR_DIR)
        archivist = ArchivistAgent(INDICES_DIR, projector_path=PROJECTOR_PATH)

        print(f"Loading MedSigLIP for full-image retrieval...")
        embed_model = SiglipVisionModel.from_pretrained(EMBED_MODEL_ID).to(device).eval()
        embed_processor = AutoProcessor.from_pretrained(EMBED_MODEL_ID)

        df = pd.read_csv(VAL_CSV)
        cache_data = []

        print(f"🚀 Processing {len(df)} images...")
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
                print(f"⚠️ Error on {case_id}: {e}")

        del embed_model
        torch.cuda.empty_cache()

        cache_df = pd.DataFrame(cache_data)
        cache_df.to_csv(CACHE_FILE, index=False)
        with open(MATCH_CACHE_FILE, "w") as f:
            json.dump(match_cache, f)
        print(f"✅ Phase 1 complete: {len(cache_df)} cases cached")

    print(f"   {len(cache_df)} cases, {len(match_cache)} match records")

    # ================================================================
    # PHASE 2: ENSEMBLE + VLM NARRATIVE
    # ================================================================
    print(f"\n⚡ Phase 2: Ensemble + VLM for {len(cache_df)} cases...")
    anthropologist = LMStudioAnthropologistAgent(
        api_url=LM_STUDIO_URL, model_id=LM_MODEL_ID
    )
    ensemble = EnsembleAgent()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_case = {
            executor.submit(
                process_vlm_request, anthropologist, ensemble, row, match_cache
            ): row["case_id"]
            for _, row in cache_df.iterrows()
        }
        for future in tqdm(as_completed(future_to_case), total=len(cache_df)):
            try:
                results.append(future.result())
            except Exception as e:
                pass

    if not results:
        print("❌ No results collected")
        return

    res_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "final_metrics.csv")
    res_df.to_csv(output_path, index=False)

    # ================================================================
    # COMPREHENSIVE REPORT
    # ================================================================
    n = len(res_df)
    gt = res_df["GT_Months"].values

    print("\n" + "#" * 70)
    print("  📊 CHRONOS-MSK FULL PIPELINE REPORT")
    print("#" * 70)

    # --- All prediction sources ---
    print(f"\n  📏 ALL PREDICTION SOURCES (MAE in months, n={n})")
    print(f"  {'─'*65}")
    print(f"  {'Source':<30s} {'MAE':>10s} {'Median AE':>12s} {'Within±12m':>12s}")
    print(f"  {'─'*65}")

    sources = [
        ("Regressor (Agent 5)", "AE_Regressor"),
        ("Ensemble (Reg+Arch)", "AE_Ensemble"),
        ("VLM Clamped (±12m)", "AE_VLM_Clamped"),
        ("VLM Unclamped (raw)", "AE_VLM_Unclamped"),
        ("Archivist Only", "AE_Archivist"),
    ]

    for name, col in sources:
        vals = res_df[col].dropna().values
        if len(vals) > 0:
            mae = np.mean(vals)
            median = np.median(vals)
            w12 = 100 * (vals <= 12).sum() / len(vals)
            print(f"  {name:<30s} {mae:>9.2f}m {median:>11.2f}m {w12:>11.1f}%")

    # --- VLM behavior analysis ---
    print(f"\n  🧠 VLM BEHAVIOR ANALYSIS")
    print(f"  {'─'*65}")

    vlm_clamped = res_df["VLM_Clamped_Months"].values
    vlm_unclamped = res_df["VLM_Unclamped_Months"].values
    reg_vals = res_df["Reg_Months"].values

    vlm_deviation = np.abs(vlm_unclamped - reg_vals)
    clamped_count = (np.abs(vlm_unclamped - vlm_clamped) > 0.1).sum()

    print(f"  VLM mean deviation from regressor: {vlm_deviation.mean():.2f}m")
    print(f"  VLM median deviation:              {np.median(vlm_deviation):.2f}m")
    print(f"  Cases where VLM was clamped:       {clamped_count} ({100*clamped_count/n:.1f}%)")
    print(f"  Cases with >12m deviation:          {(vlm_deviation > 12).sum()} "
          f"({100*(vlm_deviation > 12).sum()/n:.1f}%)")

    # Direction analysis
    vlm_higher = (vlm_unclamped > reg_vals + 0.5).sum()
    vlm_lower = (vlm_unclamped < reg_vals - 0.5).sum()
    vlm_agree = n - vlm_higher - vlm_lower
    print(f"\n  VLM tendency:")
    print(f"    Pushes age UP:    {vlm_higher} ({100*vlm_higher/n:.1f}%)")
    print(f"    Pushes age DOWN:  {vlm_lower} ({100*vlm_lower/n:.1f}%)")
    print(f"    Agrees (±0.5m):   {vlm_agree} ({100*vlm_agree/n:.1f}%)")

    # Does VLM deviation correlate with regressor error?
    reg_errors = res_df["AE_Regressor"].values
    dev_corr, dev_p = stats.pearsonr(vlm_deviation, reg_errors)
    print(f"\n  VLM deviation ↔ Regressor error: r={dev_corr:+.4f} (p={dev_p:.2e})")
    if dev_corr > 0.1:
        print(f"  ✅ VLM deviates MORE when regressor is wrong (useful signal!)")
    elif dev_corr < -0.1:
        print(f"  ❌ VLM deviates MORE when regressor is right (counterproductive)")
    else:
        print(f"  ⚠️ VLM deviation is uncorrelated with regressor error")

    # --- VLM improvement analysis ---
    print(f"\n  📊 VLM IMPACT (Clamped)")
    print(f"  {'─'*65}")

    ae_vlm = res_df["AE_VLM_Clamped"].values
    ae_reg = res_df["AE_Regressor"].values

    improved = (ae_vlm < ae_reg).sum()
    degraded = (ae_vlm > ae_reg).sum()
    tied = n - improved - degraded

    print(f"  VLM improved regressor: {improved} ({100*improved/n:.1f}%)")
    print(f"  VLM degraded regressor: {degraded} ({100*degraded/n:.1f}%)")
    print(f"  Tied:                   {tied} ({100*tied/n:.1f}%)")

    if improved > 0:
        imp_amount = (ae_reg[ae_vlm < ae_reg] - ae_vlm[ae_vlm < ae_reg]).mean()
        print(f"  Mean improvement when better: {imp_amount:.2f}m")
    if degraded > 0:
        deg_amount = (ae_vlm[ae_vlm > ae_reg] - ae_reg[ae_vlm > ae_reg]).mean()
        print(f"  Mean degradation when worse:  {deg_amount:.2f}m")

    # --- By confidence tier ---
    print(f"\n  🔍 VLM IMPACT BY CONFIDENCE")
    print(f"  {'─'*70}")
    print(f"  {'Conf':<10s} {'N':>5s} {'Reg MAE':>9s} {'VLM MAE':>9s} "
          f"{'Δ':>7s} {'Improved':>10s} {'Degraded':>10s}")
    print(f"  {'─'*70}")

    for conf in ["HIGH", "MODERATE", "LOW", "CAUTION"]:
        subset = res_df[res_df["Confidence"] == conf]
        if len(subset) > 0:
            reg_m = subset["AE_Regressor"].mean()
            vlm_m = subset["AE_VLM_Clamped"].mean()
            imp = (subset["AE_VLM_Clamped"] < subset["AE_Regressor"]).sum()
            deg = (subset["AE_VLM_Clamped"] > subset["AE_Regressor"]).sum()
            print(f"  {conf:<10s} {len(subset):>5d} {reg_m:>8.2f}m {vlm_m:>8.2f}m "
                  f"{vlm_m-reg_m:>+6.2f}m {imp:>10d} {deg:>10d}")

    # --- VLM Flag distribution ---
    print(f"\n  🏷️ VLM FLAG DISTRIBUTION")
    print(f"  {'─'*50}")
    flags = res_df["VLM_Flag"].value_counts().head(10)
    for flag, count in flags.items():
        print(f"  {str(flag)[:40]:<40s} {count:>6d}")

    # --- Age-stratified ---
    print(f"\n  📊 AGE-STRATIFIED COMPARISON")
    print(f"  {'─'*70}")
    print(f"  {'Range':<8s} {'N':>5s} {'Reg':>8s} {'Ens':>8s} {'VLM':>8s} "
          f"{'Arch':>8s} {'Best':>10s}")
    print(f"  {'─'*70}")

    for lo, hi, label in [(0, 60, "0-5y"), (60, 120, "5-10y"),
                           (120, 180, "10-15y"), (180, 228, "15-19y")]:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() > 0:
            sub = res_df[mask]
            reg_m = sub["AE_Regressor"].mean()
            ens_m = sub["AE_Ensemble"].mean()
            vlm_m = sub["AE_VLM_Clamped"].mean()
            arch_m = sub["AE_Archivist"].dropna().mean()

            best_val = min(reg_m, ens_m, vlm_m)
            if best_val == reg_m:
                best = "Regressor"
            elif best_val == ens_m:
                best = "Ensemble"
            else:
                best = "VLM"

            print(f"  {label:<8s} {mask.sum():>5d} {reg_m:>7.2f}m {ens_m:>7.2f}m "
                  f"{vlm_m:>7.2f}m {arch_m:>7.2f}m {best:>10s}")

    # --- Sample narratives ---
    print(f"\n  📝 SAMPLE VLM NARRATIVES (5 cases)")
    print(f"  {'─'*70}")

    # Show 5 diverse cases
    sample_indices = []
    # Best case
    best_idx = res_df["AE_Ensemble"].idxmin()
    sample_indices.append(best_idx)
    # Worst case
    worst_idx = res_df["AE_Ensemble"].idxmax()
    sample_indices.append(worst_idx)
    # VLM helped most
    vlm_help = (res_df["AE_Regressor"] - res_df["AE_VLM_Clamped"])
    if vlm_help.max() > 0:
        sample_indices.append(vlm_help.idxmax())
    # VLM hurt most
    if vlm_help.min() < 0:
        sample_indices.append(vlm_help.idxmin())
    # Random middle case
    mid = res_df.iloc[len(res_df)//2].name
    sample_indices.append(mid)

    for idx in sample_indices[:5]:
        row = res_df.loc[idx]
        print(f"\n  Case {row['ID']}:")
        print(f"    GT: {row['GT_Months']:.0f}m | Reg: {row['Reg_Months']:.0f}m | "
              f"Ens: {row['Ensemble_Months']:.0f}m | VLM: {row['VLM_Clamped_Months']:.0f}m")
        print(f"    Confidence: {row['Confidence']} | Matches: {row['Match_Ages']}")
        print(f"    VLM Flag: {row['VLM_Flag']}")
        if row.get('VLM_Analysis'):
            print(f"    Analysis: {str(row['VLM_Analysis'])[:150]}...")
        if row.get('VLM_Reasoning'):
            print(f"    Reasoning: {str(row['VLM_Reasoning'])[:150]}...")

    # --- Final Summary ---
    best_source = "Regressor"
    best_mae = res_df["AE_Regressor"].mean()

    for name, col in [("Ensemble", "AE_Ensemble"), ("VLM Clamped", "AE_VLM_Clamped")]:
        m = res_df[col].mean()
        if m < best_mae:
            best_mae = m
            best_source = name

    print(f"\n{'#'*70}")
    print(f"  ✅ EVALUATION COMPLETE")
    print(f"  Best source: {best_source} (MAE={best_mae:.2f}m)")
    print(f"  Regressor:   {res_df['AE_Regressor'].mean():.2f}m")
    print(f"  Ensemble:    {res_df['AE_Ensemble'].mean():.2f}m")
    print(f"  VLM Clamped: {res_df['AE_VLM_Clamped'].mean():.2f}m")
    print(f"  Archivist:   {res_df['AE_Archivist'].dropna().mean():.2f}m")

    r, _ = stats.pearsonr(gt, res_df["Ensemble_Months"].values)
    within12 = 100 * (res_df["AE_Ensemble"] <= 12).sum() / n
    print(f"  Pearson r:   {r:.4f}")
    print(f"  Within ±12m: {within12:.1f}%")
    print(f"\n  Results: {output_path}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
