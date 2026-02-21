"""
eval_embedding_fix.py
=====================
Evaluates the pipeline with FULL IMAGE embeddings for retrieval.
Uses SOTA projector and distance-calibrated ensemble.
"""

import os
import json
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

# --- CONFIG ---
VAL_CSV = "data/boneage_val.csv"
IMAGE_DIR = "/mnt/e/chronos-msk/data/RSNA Bone Age+Anatomical ROIS/RSNA_val/images"
OUTPUT_DIR = "evaluation_results_embedding_fix"
CACHE_FILE = os.path.join(OUTPUT_DIR, "cache.csv")
MATCH_CACHE_FILE = os.path.join(OUTPUT_DIR, "matches.json")

INDICES_DIR = "indices_projected_256d"
PROJECTOR_PATH = "weights/projector_sota.pth"
SCOUT_WEIGHTS = "weights/best_scout.pt"
SVM_WEIGHTS = "weights/radiologist_head.pkl"
REGRESSOR_DIR = "weights/medsiglip_sota"
EMBED_MODEL_ID = "google/medsiglip-448"

AVAILABLE_RACES = ["Asian", "Caucasian", "Hispanic", "Black"]


def letterbox_resize(image, size=448):
    """Resizes image preserving aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized
    return padded


def get_full_image_embedding(img_path, embed_model, processor, device):
    """Embed full image with MedSigLIP (matches index domain)."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = letterbox_resize(img_rgb, size=448)
    inputs = processor(images=img_resized, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = embed_model(**inputs).pooler_output
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # PHASE 1: LOCAL INFERENCE
    # ================================================================
    if os.path.exists(CACHE_FILE) and os.path.exists(MATCH_CACHE_FILE):
        print(f"⚡ Loading cache from {CACHE_FILE}...")
        cache_df = pd.read_csv(CACHE_FILE)
        with open(MATCH_CACHE_FILE) as f:
            match_cache = json.load(f)
        print(f"   {len(cache_df)} cases, {len(match_cache)} match records")
    else:
        print("🐢 Running Phase 1 with FULL IMAGE embeddings + SOTA projector...")

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

        print(f"🚀 Processing {len(df)} images...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            case_id = str(row["ID"])
            img_path = os.path.join(IMAGE_DIR, f"{case_id}.png")
            if not os.path.exists(img_path):
                continue

            sex_str = "Male" if row["Male"] else "Female"
            try:
                # Agent 1+2: CROP for staging
                crop_bgr = scout.predict(img_path)
                stage, _ = radiologist.predict(crop_bgr)

                # Agent 3: FULL IMAGE embedding for retrieval
                full_embedding = get_full_image_embedding(
                    img_path, embed_model, embed_processor, device
                )
                if full_embedding is None:
                    continue

                all_candidates = []
                for race in AVAILABLE_RACES:
                    race_matches = archivist.retrieve(
                        full_embedding, sex_str, race, top_k=3
                    )
                    all_candidates.extend(race_matches)
                all_candidates.sort(key=lambda x: x.get("distance", 999.0))
                best_matches = all_candidates[:5]

                # Agent 5: Regressor
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

        # Free GPU
        del embed_model
        torch.cuda.empty_cache()

        cache_df = pd.DataFrame(cache_data)
        cache_df.to_csv(CACHE_FILE, index=False)
        with open(MATCH_CACHE_FILE, "w") as f:
            json.dump(match_cache, f)
        print(f"✅ Phase 1 complete: {len(cache_df)} cases cached")

    # ================================================================
    # PHASE 2: ENSEMBLE SCORING
    # ================================================================
    print(f"\n⚡ Phase 2: Scoring {len(cache_df)} cases...")
    ensemble = EnsembleAgent()

    results = []
    for _, row in tqdm(cache_df.iterrows(), total=len(cache_df)):
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

    res_df = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR, "embedding_fix_metrics.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"💾 Raw results saved to {output_csv}")

    # ================================================================
    # FULL REPORT
    # ================================================================
    gt = res_df["gt_months"].values
    pred = res_df["final_age_months"].values
    ae = res_df["ae_ensemble"].values
    reg_ae = res_df["ae_regressor"].values
    n = len(res_df)

    r, p_val = stats.pearsonr(gt, pred)

    arch_ae_all = res_df["ae_archivist"].dropna().values
    arch_mae = np.mean(arch_ae_all) if len(arch_ae_all) > 0 else float("nan")

    print("\n" + "#" * 70)
    print("  📊 CHRONOS-MSK — FULL PIPELINE METRICS")
    print("#" * 70)

    # --- Core Accuracy ---
    print(f"\n  📏 CORE ACCURACY (n={n})")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<35s} {'Value':>20s}")
    print(f"  {'─'*60}")
    print(f"  {'Ensemble MAE':<35s} {np.mean(ae):>15.2f} months")
    print(f"  {'Regressor-only MAE':<35s} {np.mean(reg_ae):>15.2f} months")
    print(f"  {'Archivist-only MAE':<35s} {arch_mae:>15.2f} months")
    print(f"  {'Ensemble Median AE':<35s} {np.median(ae):>15.2f} months")
    print(f"  {'RMSE':<35s} {np.sqrt(np.mean(ae**2)):>15.2f} months")
    print(f"  {'Pearson r':<35s} {r:>15.4f}")
    print(f"  {'R²':<35s} {r**2:>15.4f}")
    print(f"  {'p-value':<35s} {p_val:>15.2e}")

    delta = np.mean(ae) - np.mean(reg_ae)
    if delta < -0.01:
        print(f"\n  ✅ ENSEMBLE IMPROVES OVER REGRESSOR BY {-delta:.2f} MONTHS")
    elif delta > 0.01:
        print(f"\n  ⚠️ Ensemble is {delta:.2f} months WORSE than regressor alone")
    else:
        print(f"\n  ➡️ Ensemble matches regressor (Δ={delta:.4f})")

    # --- Threshold Accuracy ---
    print(f"\n  🎯 THRESHOLD ACCURACY")
    print(f"  {'─'*60}")
    print(f"  {'Threshold':<20s} {'Ensemble':>12s} {'Regressor':>12s} {'Δ':>10s}")
    print(f"  {'─'*60}")
    for t in [3, 6, 9, 12, 18, 24, 36]:
        ens_pct = 100 * (ae <= t).sum() / n
        reg_pct = 100 * (reg_ae <= t).sum() / n
        d = ens_pct - reg_pct
        print(f"  Within ±{t:2d}mo       {ens_pct:>10.1f}%  {reg_pct:>10.1f}%  {d:>+8.1f}%")

    # --- Blending Analysis ---
    print(f"\n  ⚖️ BLENDING ANALYSIS")
    print(f"  {'─'*60}")
    blended = res_df[res_df["method"] == "blended"]
    regonly = res_df[res_df["method"] == "regressor_only"]
    print(f"  Cases blended:        {len(blended)} ({100*len(blended)/n:.1f}%)")
    print(f"  Cases regressor-only: {len(regonly)} ({100*len(regonly)/n:.1f}%)")

    if len(blended) > 0:
        blended_ae = blended["ae_ensemble"].values
        blended_reg_ae = blended["ae_regressor"].values
        b_delta = np.mean(blended_ae) - np.mean(blended_reg_ae)

        print(f"\n  Blended cases performance:")
        print(f"    Ensemble MAE:  {np.mean(blended_ae):.2f}m")
        print(f"    Regressor MAE: {np.mean(blended_reg_ae):.2f}m")
        print(f"    Δ:             {b_delta:+.2f}m")

        improved = (blended["ae_ensemble"] < blended["ae_regressor"]).sum()
        degraded = (blended["ae_ensemble"] > blended["ae_regressor"]).sum()
        tied = (blended["ae_ensemble"] == blended["ae_regressor"]).sum()
        print(f"    Improved: {improved}/{len(blended)} ({100*improved/len(blended):.1f}%)")
        print(f"    Degraded: {degraded}/{len(blended)} ({100*degraded/len(blended):.1f}%)")
        print(f"    Tied:     {tied}/{len(blended)} ({100*tied/len(blended):.1f}%)")

        if "blend_weight" in blended.columns:
            weights = blended["blend_weight"]
            print(f"\n  Blend weight distribution:")
            print(f"    Mean:   {weights.mean():.4f}")
            print(f"    Median: {weights.median():.4f}")
            print(f"    Min:    {weights.min():.4f}")
            print(f"    Max:    {weights.max():.4f}")

    # --- Confidence Calibration ---
    print(f"\n  🔍 CONFIDENCE CALIBRATION")
    print(f"  {'─'*70}")
    print(f"  {'Confidence':<12s} {'N':>6s} {'MAE':>10s} {'Median AE':>12s} "
          f"{'Within±12m':>12s} {'Reg MAE':>10s}")
    print(f"  {'─'*70}")

    conf_maes = []
    for conf in ["HIGH", "MODERATE", "LOW", "CAUTION", "REVIEW"]:
        subset = res_df[res_df["confidence"] == conf]
        if len(subset) > 0:
            conf_mae = subset["ae_ensemble"].mean()
            conf_median = subset["ae_ensemble"].median()
            within_12 = 100 * (subset["ae_ensemble"] <= 12).sum() / len(subset)
            reg_mae_conf = subset["ae_regressor"].mean()
            conf_maes.append(conf_mae)
            print(f"  {conf:<12s} {len(subset):>6d} {conf_mae:>9.2f}m "
                  f"{conf_median:>11.2f}m {within_12:>11.1f}% "
                  f"{reg_mae_conf:>9.2f}m")

    if len(conf_maes) >= 2:
        is_monotonic = all(
            conf_maes[i] <= conf_maes[i + 1] + 0.5
            for i in range(len(conf_maes) - 1)
        )
        if is_monotonic:
            print(f"\n  ✅ PROPERLY CALIBRATED")
        else:
            print(f"\n  ⚠️ Not perfectly monotonic: {[f'{m:.2f}' for m in conf_maes]}")

    conf_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2, "CAUTION": 3, "REVIEW": 4}
    res_df["conf_rank"] = res_df["confidence"].map(conf_order)
    valid_conf = res_df.dropna(subset=["conf_rank"])
    if len(valid_conf) > 10:
        conf_corr, _ = stats.pearsonr(
            valid_conf["conf_rank"], valid_conf["ae_ensemble"]
        )
        print(f"  Confidence ↔ Error correlation: {conf_corr:+.4f}")

    # --- Agreement ---
    print(f"\n  🤝 REGRESSOR ↔ ARCHIVIST AGREEMENT")
    print(f"  {'─'*60}")
    if "agreement_months" in res_df.columns:
        agreement = res_df["agreement_months"].dropna()
        if len(agreement) > 0:
            print(f"  Mean gap:        {agreement.mean():.2f} months")
            print(f"  Median gap:      {agreement.median():.2f} months")
            print(f"  Within 12m:      {(agreement <= 12).sum()} "
                  f"({100*(agreement <= 12).sum()/len(agreement):.1f}%)")
            print(f"  Within 24m:      {(agreement <= 24).sum()} "
                  f"({100*(agreement <= 24).sum()/len(agreement):.1f}%)")
            print(f"  Beyond 48m:      {(agreement > 48).sum()} "
                  f"({100*(agreement > 48).sum()/len(agreement):.1f}%)")

    # --- Distance ---
    print(f"\n  📏 RETRIEVAL DISTANCE ANALYSIS")
    print(f"  {'─'*60}")
    if "avg_retrieval_distance" in res_df.columns:
        dists = res_df["avg_retrieval_distance"].dropna()
        if len(dists) > 0:
            print(f"  Mean:    {dists.mean():.4f}")
            print(f"  Median:  {dists.median():.4f}")
            print(f"  P10:     {dists.quantile(0.10):.4f}")
            print(f"  P25:     {dists.quantile(0.25):.4f}")
            print(f"  P75:     {dists.quantile(0.75):.4f}")
            print(f"  P90:     {dists.quantile(0.90):.4f}")

            arch_errors = res_df["ae_archivist"].dropna()
            common_idx = arch_errors.index.intersection(dists.index)
            if len(common_idx) > 10:
                d_corr, d_p = stats.pearsonr(
                    dists.loc[common_idx], arch_errors.loc[common_idx]
                )
                print(f"\n  Distance ↔ Archivist Error: r={d_corr:+.4f} (p={d_p:.2e})")
                if d_corr > 0.15:
                    print(f"  ✅ Distance is a meaningful quality signal!")
                else:
                    print(f"  ⚠️ Distance weakly correlated with error")

    # --- Age-Stratified ---
    print(f"\n  📊 AGE-STRATIFIED MAE")
    print(f"  {'─'*75}")
    print(f"  {'Range':<12s} {'N':>5s} {'Ensemble':>10s} {'Regressor':>10s} "
          f"{'Archivist':>10s} {'Δ(E-R)':>8s} {'Winner':>10s}")
    print(f"  {'─'*75}")

    for lo, hi, label in [
        (0, 24, "0-2y"), (24, 60, "2-5y"), (60, 96, "5-8y"),
        (96, 120, "8-10y"), (120, 144, "10-12y"), (144, 168, "12-14y"),
        (168, 192, "14-16y"), (192, 228, "16-19y"),
    ]:
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() > 0:
            e_mae = np.mean(ae[mask])
            r_mae = np.mean(reg_ae[mask])
            a_vals = res_df.loc[mask, "ae_archivist"].dropna()
            a_mae = a_vals.mean() if len(a_vals) > 0 else float("nan")
            d = e_mae - r_mae
            winner = "Ensemble" if d < -0.1 else "Regressor" if d > 0.1 else "Tie"
            print(f"  {label:<12s} {mask.sum():>5d} {e_mae:>9.2f}m "
                  f"{r_mae:>9.2f}m {a_mae:>9.2f}m {d:>+7.2f}m {winner:>10s}")

    # --- Sex-Stratified ---
    print(f"\n  👤 SEX-STRATIFIED MAE")
    print(f"  {'─'*60}")
    for sex in ["Male", "Female"]:
        mask = res_df["sex_str"] == sex
        if mask.sum() > 0:
            sex_ens = np.mean(ae[mask])
            sex_reg = np.mean(reg_ae[mask])
            d = sex_ens - sex_reg
            print(f"  {sex:<10s} (n={mask.sum():4d}): "
                  f"Ensemble={sex_ens:.2f}m, Regressor={sex_reg:.2f}m, "
                  f"Δ={d:+.2f}m")

    # --- Error Distribution ---
    print(f"\n  📈 ERROR DISTRIBUTION (Ensemble)")
    print(f"  {'─'*60}")
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(ae, p)
        print(f"  P{p:<3d}:  {val:>8.2f} months")

    # --- Worst Cases ---
    print(f"\n  🔴 WORST 10 CASES")
    print(f"  {'─'*80}")
    print(f"  {'ID':<10s} {'GT':>7s} {'Ens':>7s} {'Reg':>7s} {'Arch':>7s} "
          f"{'AE':>7s} {'Conf':<10s} {'Method':<15s}")
    print(f"  {'─'*80}")

    worst = res_df.nlargest(10, "ae_ensemble")
    for _, row in worst.iterrows():
        arch_p = row.get("archivist_pred", None)
        arch_str = f"{arch_p:.0f}m" if arch_p and not np.isnan(arch_p) else "N/A"
        print(f"  {str(row['ID']):<10s} {row['gt_months']:>6.0f}m "
              f"{row['final_age_months']:>6.0f}m {row['regressor_pred']:>6.0f}m "
              f"{arch_str:>7s} {row['ae_ensemble']:>6.1f}m "
              f"{row['confidence']:<10s} {row['method']:<15s}")

    # --- Best Cases ---
    print(f"\n  🟢 BEST 10 CASES")
    print(f"  {'─'*80}")
    print(f"  {'ID':<10s} {'GT':>7s} {'Ens':>7s} {'Reg':>7s} "
          f"{'AE':>7s} {'Conf':<10s} {'Method':<15s}")
    print(f"  {'─'*80}")

    best = res_df.nsmallest(10, "ae_ensemble")
    for _, row in best.iterrows():
        print(f"  {str(row['ID']):<10s} {row['gt_months']:>6.0f}m "
              f"{row['final_age_months']:>6.0f}m {row['regressor_pred']:>6.0f}m "
              f"{row['ae_ensemble']:>6.1f}m "
              f"{row['confidence']:<10s} {row['method']:<15s}")

    # --- Grid Search ---
    print(f"\n  🔎 OPTIMAL BLENDING GRID SEARCH")
    print(f"  {'─'*60}")

    best_mae = np.mean(reg_ae)
    best_params = None

    for dist_thresh in np.arange(0.08, 0.50, 0.02):
        for max_w in np.arange(0.05, 0.50, 0.05):
            for age_lo in [0, 60, 80, 96]:
                for age_hi in [168, 180, 192, 228]:
                    test_preds = []
                    for _, row in res_df.iterrows():
                        reg = row["regressor_pred"]
                        arch = row.get("archivist_pred", None)
                        dist = row.get("avg_retrieval_distance", 999)
                        gt_val = row["gt_months"]

                        if (arch is not None
                            and not np.isnan(arch)
                            and dist <= dist_thresh
                            and age_lo <= reg < age_hi):
                            w = max_w * (1.0 - dist / dist_thresh)
                            p = (1 - w) * reg + w * arch
                        else:
                            p = reg

                        test_preds.append(abs(p - gt_val))

                    test_mae = np.mean(test_preds)
                    if test_mae < best_mae - 0.001:
                        best_mae = test_mae
                        best_params = {
                            "dist_threshold": round(float(dist_thresh), 2),
                            "max_weight": round(float(max_w), 2),
                            "age_range": f"{age_lo}-{age_hi}m",
                        }

    if best_params:
        improvement = np.mean(reg_ae) - best_mae
        print(f"  Best ensemble MAE:  {best_mae:.4f} months")
        print(f"  Regressor-only MAE: {np.mean(reg_ae):.4f} months")
        print(f"  Improvement:        {improvement:.4f} months")
        print(f"  Optimal params:     {best_params}")
    else:
        print(f"  No configuration beats regressor alone ({np.mean(reg_ae):.4f})")

    # --- Markdown Export ---
    markdown = f"""# Chronos-MSK Pipeline Evaluation (SOTA Projector)

## Core Metrics (n={n})

| Metric | Value |
|---|---|
| **Ensemble MAE** | **{np.mean(ae):.2f} months ({np.mean(ae)/12:.2f} years)** |
| Regressor-only MAE | {np.mean(reg_ae):.2f} months |
| Archivist-only MAE | {arch_mae:.2f} months |
| Median AE | {np.median(ae):.2f} months |
| RMSE | {np.sqrt(np.mean(ae**2)):.2f} months |
| Pearson r | {r:.4f} |
| R² | {r**2:.4f} |
| Within ±6 months | {100*(ae<=6).sum()/n:.1f}% |
| Within ±12 months | {100*(ae<=12).sum()/n:.1f}% |
| Within ±24 months | {100*(ae<=24).sum()/n:.1f}% |

## Models Used

| Model | Role |
|---|---|
| MedSigLIP-448 | Vision backbone, regression, embedding |
| MedGemma 1.5 4B-IT | Clinical narrative generation |
| Custom YOLO | Anatomical region detection |
| SOTA Projector | 1152→256D metric learning |
"""

    md_path = os.path.join(OUTPUT_DIR, "competition_metrics.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"\n📝 Markdown saved to {md_path}")

    # --- Final ---
    print("\n" + "#" * 70)
    ens_mae = np.mean(ae)
    within_12 = 100 * (ae <= 12).sum() / n
    print(f"  ✅ EVALUATION COMPLETE")
    print(f"  Ensemble MAE: {ens_mae:.2f}m | Pearson r: {r:.4f} | "
          f"Within ±12m: {within_12:.1f}%")
    if best_params:
        print(f"  Optimal blending: MAE={best_mae:.4f}m "
              f"(Δ={np.mean(reg_ae)-best_mae:+.4f}m)")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()