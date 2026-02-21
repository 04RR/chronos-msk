"""
agents/agent4_vlm_client.py
============================
Fixed VLM Client with output clamping and regressor-anchored prompt.
"""

import requests
import base64
import json
import os
import io
import re
import numpy as np
from PIL import Image


class LMStudioAnthropologistAgent:
    """
    VLM-based reasoning agent for bone age estimation.
    Acts as a REFINEMENT layer on top of the regressor, NOT a replacement.
    """

    # Maximum months the VLM is allowed to adjust the regressor prediction
    MAX_DELTA_MONTHS = 12.0

    # Valid age range for pediatric bone age (0 to 19 years)
    MIN_AGE_MONTHS = 0.0
    MAX_AGE_MONTHS = 228.0

    def __init__(
        self,
        api_url="http://localhost:1234/v1/chat/completions",
        model_id="medgemma-1.5-4b-it",
    ):
        self.api_url = api_url
        self.model_id = model_id
        print(f"🧠 VLM Client: Configured for {api_url} ({model_id})")

    def _encode_image(self, image_path):
        """Resizes and encodes image to base64."""
        try:
            if not os.path.exists(image_path):
                return ""
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                max_size = 768
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Image Encode Error: {e}")
            return ""

    def _extract_age_via_regex(self, text, default_age):
        """Fallback regex extraction for final_age_months."""
        try:
            # Pattern 1: JSON-like key "final_age_months": 123
            pattern1 = r'"final_age_months"\s*:\s*(\d+(?:\.\d+)?)'
            match = re.search(pattern1, text)
            if match:
                return float(match.group(1))

            # Pattern 2: Natural language "final age is 123 months"
            pattern2 = r"final age (?:is |of )?(\d+(?:\.\d+)?)"
            match = re.search(pattern2, text, re.IGNORECASE)
            if match:
                return float(match.group(1))

            return float(default_age)
        except:
            return float(default_age)

    def _clean_and_parse_json(self, raw_text, default_age):
        """Parse VLM JSON output with multiple fallback strategies."""
        try:
            clean_text = raw_text.replace("<unused94>", "").replace(
                "<start_of_turn>", ""
            )

            # Try code block extraction first
            code_pattern = r"```(?:json)?\s*(.*?)\s*```"
            code_match = re.search(code_pattern, clean_text, re.DOTALL)
            if code_match:
                clean_text = code_match.group(1)

            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}")

            if start_idx == -1 or end_idx == -1:
                extracted_age = self._extract_age_via_regex(
                    clean_text, default_age
                )
                return {
                    "visual_analysis": "JSON Parse Failed (Regex Recovery)",
                    "conflict_resolution": raw_text[:200],
                    "final_age_months": extracted_age,
                    "flag": "⚠️ Regex Recovery",
                }

            json_str = clean_text[start_idx : end_idx + 1]
            return json.loads(json_str)

        except Exception as e:
            extracted_age = self._extract_age_via_regex(raw_text, default_age)
            return {
                "visual_analysis": "Parse Error",
                "conflict_resolution": f"Raw: {raw_text[:100]}...",
                "final_age_months": extracted_age,
                "flag": f"⚠️ Parse Error: {str(e)}",
            }

    def _clamp_age(self, vlm_age, reg_age):
        """
        Clamp VLM output to be within ±MAX_DELTA of regressor,
        and within valid pediatric range.
        """
        # Step 1: Clamp to valid range
        clamped = np.clip(vlm_age, self.MIN_AGE_MONTHS, self.MAX_AGE_MONTHS)

        # Step 2: Clamp to regressor ± delta
        lower = max(self.MIN_AGE_MONTHS, reg_age - self.MAX_DELTA_MONTHS)
        upper = min(self.MAX_AGE_MONTHS, reg_age + self.MAX_DELTA_MONTHS)
        clamped = np.clip(clamped, lower, upper)

        return round(float(clamped), 2)

    def analyze(self, image_path, sex, race, stage, matches, reg_age_months):
        """
        Main analysis method.
        The VLM acts as a REFINEMENT on the regressor, not a replacement.
        """
        # ====== INPUT VALIDATION ======
        reg_age_months = float(reg_age_months)
        if reg_age_months < self.MIN_AGE_MONTHS or reg_age_months > self.MAX_AGE_MONTHS:
            return {
                "final_age_months": np.clip(
                    reg_age_months, self.MIN_AGE_MONTHS, self.MAX_AGE_MONTHS
                ),
                "flag": f"⚠️ Regressor out of range ({reg_age_months:.1f}m)",
            }

        # ====== BUILD EVIDENCE SUMMARY ======
        archivist_summary = "No visual matches available."
        trust_level = "LOW"

        if matches and len(matches) > 0:
            valid_ages = [
                m.get("age_months", -1)
                for m in matches
                if m.get("age_months", -1) > 0
            ]
            distances = [m.get("distance", 999) for m in matches]

            if valid_ages:
                avg_match_months = np.mean(valid_ages)
                avg_distance = np.mean(distances)

                # Trust thresholds (empirically calibrated)
                if avg_distance < 0.10:
                    trust_level = "HIGH"
                elif avg_distance < 0.15:
                    trust_level = "MEDIUM"
                else:
                    trust_level = "LOW"

                age_strs = [f"{m:.0f}m" for m in valid_ages]
                archivist_summary = (
                    f"Retrieved {len(valid_ages)} Visual Twins.\n"
                    f"   Ages: {', '.join(age_strs)}\n"
                    f"   Peer Average: {avg_match_months:.1f} months\n"
                    f"   Trust: {trust_level} (avg dist: {avg_distance:.4f})"
                )

        # ====== PROMPT (Regressor-Anchored) ======
        reg_years = reg_age_months / 12.0
        delta = self.MAX_DELTA_MONTHS
        lower_bound = max(0, reg_age_months - delta)
        upper_bound = min(228, reg_age_months + delta)

        system_instruction = (
            "You are a Forensic Anthropologist specializing in skeletal age estimation. "
            "You must output ONLY valid JSON. "
            f"The valid age range is {self.MIN_AGE_MONTHS:.0f}-{self.MAX_AGE_MONTHS:.0f} months."
        )

        user_content = (
            f"[PRIMARY EVIDENCE — Your output MUST be within ±{delta:.0f} months of this value]\n"
            f"Expert Regressor Estimate: {reg_age_months:.1f} months ({reg_years:.1f} years)\n"
            f"Valid output range: {lower_bound:.0f} to {upper_bound:.0f} months\n"
            f"\n"
            f"[SECONDARY EVIDENCE — For confirmation only]\n"
            f"Demographics: {sex}, {race}\n"
            f"Visual Matches (Trust: {trust_level}):\n"
            f"{archivist_summary}\n"
            f"\n"
            f"[RULES]\n"
            f"1. Start from the Expert Regressor value ({reg_age_months:.1f} months).\n"
            f"2. Only adjust if you see STRONG visual evidence in the X-ray.\n"
            f"3. Your final_age_months MUST be between {lower_bound:.0f} and {upper_bound:.0f}.\n"
            f"4. If Visual Match trust is LOW, IGNORE them completely.\n"
            f"5. Default to the Regressor value if uncertain.\n"
            f"\n"
            f'[OUTPUT — JSON ONLY]\n'
            f'{{\n'
            f'  "visual_analysis": "what you observe in the X-ray",\n'
            f'  "adjustment_reasoning": "why you adjusted (or \'confirmed regressor\')",\n'
            f'  "final_age_months": <number between {lower_bound:.0f} and {upper_bound:.0f}>,\n'
            f'  "flag": "confidence note"\n'
            f'}}\n'
        )

        # ====== API CALL ======
        base64_img = self._encode_image(image_path)
        if not base64_img:
            return {
                "final_age_months": reg_age_months,
                "flag": "⚠️ Image Error — using regressor",
            }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "stream": False,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            if response.status_code != 200:
                return {
                    "final_age_months": reg_age_months,
                    "flag": f"⚠️ API {response.status_code} — using regressor",
                }

            data = response.json()
            raw_content = data["choices"][0]["message"]["content"]
            result = self._clean_and_parse_json(raw_content, reg_age_months)

            # ====== CRITICAL: CLAMP OUTPUT ======
            raw_vlm_age = float(result.get("final_age_months", reg_age_months))
            clamped_age = self._clamp_age(raw_vlm_age, reg_age_months)

            result["final_age_months"] = clamped_age
            result["unclamped_vlm_age"] = raw_vlm_age
            result["regressor_anchor"] = reg_age_months

            if abs(raw_vlm_age - clamped_age) > 0.1:
                result["flag"] = (
                    f"⚠️ Clamped: VLM said {raw_vlm_age:.1f}m, "
                    f"bounded to {clamped_age:.1f}m"
                )

            return result

        except Exception as e:
            return {
                "final_age_months": reg_age_months,
                "flag": "⚠️ System Error — using regressor",
            }