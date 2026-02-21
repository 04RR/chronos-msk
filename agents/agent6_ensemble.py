"""
agents/agent6_ensemble.py
=========================
Production ensemble with empirically optimized parameters.

Grid search results (1,425 RSNA validation cases):
- Optimal: dist<0.26, weight=0.10, ages 60-192m
- Improvement: 0.029 months over regressor alone
- Primary value of retrieval: explainability, not prediction
"""

import numpy as np


class EnsembleAgent:
    """
    Distance-calibrated ensemble with grid-search-optimized parameters.
    """
    
    # Optimal parameters from grid search
    DIST_THRESHOLD = 0.26
    MAX_BLEND_WEIGHT = 0.10
    BLEND_AGE_LO = 60     # 5 years
    BLEND_AGE_HI = 192    # 16 years
    
    # Performance stats from validation (for confidence reporting)
    AGE_PERFORMANCE = {
        (0, 24):    {"mae": 12.86, "within_12": 53.8, "tier": "CAUTION"},
        (24, 60):   {"mae": 9.01,  "within_12": 70.4, "tier": "MODERATE"},
        (60, 96):   {"mae": 9.85,  "within_12": 67.3, "tier": "MODERATE"},
        (96, 120):  {"mae": 10.36, "within_12": 65.1, "tier": "LOW"},
        (120, 144): {"mae": 9.97,  "within_12": 67.0, "tier": "MODERATE"},
        (144, 168): {"mae": 6.77,  "within_12": 82.4, "tier": "HIGH"},
        (168, 192): {"mae": 7.02,  "within_12": 80.2, "tier": "HIGH"},
        (192, 228): {"mae": 10.16, "within_12": 65.5, "tier": "LOW"},
    }
    
    GLOBAL_MAE = 8.81
    GLOBAL_WITHIN_12 = 73.2
    
    def __init__(self):
        print("⚖️ Ensemble Agent initialized (optimized blend)")
    
    def _get_age_stats(self, age_months):
        for (lo, hi), stats in self.AGE_PERFORMANCE.items():
            if lo <= age_months < hi:
                return stats, f"{lo // 12}-{hi // 12} years"
        return {"mae": self.GLOBAL_MAE, "within_12": self.GLOBAL_WITHIN_12, 
                "tier": "MODERATE"}, "all ages"
    
    def predict(self, reg_age_months, matches, gt_months=None):
        reg_pred = float(reg_age_months)
        
        # --- Analyze retrieval ---
        valid_matches = [
            m for m in (matches or [])
            if m.get("age_months", -1) > 0
        ]
        
        arch_pred = None
        avg_dist = None
        blend_weight = 0.0
        
        if valid_matches:
            distances = np.array([m["distance"] for m in valid_matches])
            ages = np.array([m["age_months"] for m in valid_matches])
            inv_dist = 1.0 / (distances + 1e-6)
            arch_pred = float(np.average(ages, weights=inv_dist))
            avg_dist = float(np.mean(distances))
            
            # Blend only within optimal age range and distance
            in_age_range = self.BLEND_AGE_LO <= reg_pred < self.BLEND_AGE_HI
            in_dist_range = avg_dist <= self.DIST_THRESHOLD
            
            if in_age_range and in_dist_range:
                # Linear decay: full weight at dist=0, zero at threshold
                dist_factor = 1.0 - (avg_dist / self.DIST_THRESHOLD)
                blend_weight = self.MAX_BLEND_WEIGHT * dist_factor
        
        # --- Compute final prediction ---
        if blend_weight > 0 and arch_pred is not None:
            final_pred = (1.0 - blend_weight) * reg_pred + blend_weight * arch_pred
            final_pred = np.clip(final_pred, 0.0, 228.0)
            method = "blended"
        else:
            final_pred = reg_pred
            method = "regressor_only"
        
        # --- Confidence from age-range performance ---
        age_stats, age_range_str = self._get_age_stats(reg_pred)
        confidence = age_stats["tier"]
        expected_mae = age_stats["mae"]
        within_12 = age_stats["within_12"]
        
        # --- Build explanation ---
        peer_note = ""
        if arch_pred is not None:
            agreement = abs(reg_pred - arch_pred)
            peer_note = (
                f" Atlas comparison: {arch_pred:.0f}m "
                f"({'agrees' if agreement < 12 else 'differs'}, "
                f"Δ={agreement:.0f}m)."
            )
        
        if method == "blended":
            explanation = (
                f"Bone age: {final_pred:.0f} months ({final_pred/12:.1f} years). "
                f"AI model: {reg_pred:.0f}m, refined with atlas data "
                f"(weight={blend_weight:.0%}).{peer_note} "
                f"For {age_range_str}: expected accuracy ±{expected_mae:.0f}m, "
                f"{within_12:.0f}% within ±12m."
            )
        else:
            explanation = (
                f"Bone age: {final_pred:.0f} months ({final_pred/12:.1f} years). "
                f"Based on AI regression model.{peer_note} "
                f"For {age_range_str}: expected accuracy ±{expected_mae:.0f}m, "
                f"{within_12:.0f}% within ±12m."
            )
        
        # --- Build result ---
        result = {
            "final_age_months": round(float(final_pred), 2),
            "final_age_years": round(float(final_pred) / 12.0, 1),
            "confidence": confidence,
            "expected_mae": round(expected_mae, 1),
            "within_12m_pct": round(within_12, 1),
            "age_range": age_range_str,
            "method": method,
            "explanation": explanation,
            "regressor_pred": round(reg_pred, 2),
            "archivist_pred": round(arch_pred, 2) if arch_pred is not None else None,
            "blend_weight": round(blend_weight, 4),
            "avg_retrieval_distance": round(avg_dist, 4) if avg_dist is not None else None,
            "n_matches": len(valid_matches),
        }
        
        if arch_pred is not None:
            result["agreement_months"] = round(abs(reg_pred - arch_pred), 2)
            result["match_ages"] = [round(m["age_months"], 1) for m in valid_matches]
            result["match_distances"] = [round(m["distance"], 4) for m in valid_matches]
            result["match_partitions"] = [m.get("partition", "?") for m in valid_matches]
        
        if gt_months is not None:
            gt = float(gt_months)
            result["gt_months"] = gt
            result["ae_ensemble"] = round(abs(float(final_pred) - gt), 2)
            result["ae_regressor"] = round(abs(reg_pred - gt), 2)
            if arch_pred is not None:
                result["ae_archivist"] = round(abs(arch_pred - gt), 2)
        
        return result
