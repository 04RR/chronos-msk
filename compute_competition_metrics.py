"""
Generates the complete metrics summary for the competition writeup.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats


def compute_metrics(results_csv):
    df = pd.read_csv(results_csv)

    # Auto-detect columns
    if "ae_ensemble" in df.columns:
        gt_col, pred_col, ae_col = "gt_months", "final_age_months", "ae_ensemble"
        reg_ae_col = "ae_regressor"
    elif "AE_Pipeline" in df.columns:
        gt_col, pred_col, ae_col = "GT_Months", "Pred_Months", "AE_Pipeline"
        reg_ae_col = "AE_Regressor"
    elif "AE_Ensemble" in df.columns:
        gt_col, pred_col, ae_col = "GT_Months", "Ensemble_Months", "AE_Ensemble"
        reg_ae_col = "AE_Regressor"
    else:
        print(f"Cannot identify columns: {list(df.columns)}")
        return

    gt = df[gt_col].values
    ae = df[ae_col].values
    n = len(df)

    if pred_col in df.columns:
        pred = df[pred_col].values
        r, p_val = stats.pearsonr(gt, pred)
    else:
        r, p_val = 0, 1

    print("\n" + "=" * 60)
    print("CHRONOS-MSK COMPETITION METRICS")
    print("=" * 60)

    print(f"\n  Cases: {n}")
    print(f"\n  ACCURACY")
    print(f"  {'─'*45}")
    print(f"  MAE:           {np.mean(ae):.2f} months ({np.mean(ae)/12:.2f} years)")
    print(f"  Median AE:     {np.median(ae):.2f} months")
    print(f"  RMSE:          {np.sqrt(np.mean(ae**2)):.2f} months")
    print(f"  Pearson r:     {r:.4f}")
    print(f"  R squared:     {r**2:.4f}")

    print(f"\n  THRESHOLD ACCURACY")
    print(f"  {'─'*45}")
    for t in [6, 12, 18, 24]:
        pct = 100 * (ae <= t).sum() / n
        print(f"  Within +/-{t:2d}mo: {pct:.1f}%")

    if reg_ae_col in df.columns:
        reg_mae = df[reg_ae_col].mean()
        print(f"\n  Regressor-only MAE: {reg_mae:.2f} months")

    # Export markdown
    markdown = f"""| Metric | Value |
|---|---|
| **MAE** | **{np.mean(ae):.2f} months ({np.mean(ae)/12:.2f} years)** |
| Median AE | {np.median(ae):.2f} months |
| RMSE | {np.sqrt(np.mean(ae**2)):.2f} months |
| Pearson r | {r:.4f} |
| R squared | {r**2:.4f} |
| Within +/-6 months | {100*(ae<=6).sum()/n:.1f}% |
| Within +/-12 months | {100*(ae<=12).sum()/n:.1f}% |
| Within +/-24 months | {100*(ae<=24).sum()/n:.1f}% |
"""
    md_path = os.path.join(os.path.dirname(results_csv), "metrics_table.md")
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"\n  Saved to {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    candidates = [
        "evaluation_results/embedding_fix_metrics.csv",
        "evaluation_results/final_metrics.csv",
        "evaluation_results/ensemble_metrics.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Using: {path}")
            compute_metrics(path)
            break
    else:
        print("No results file found. Run evaluation first.")

