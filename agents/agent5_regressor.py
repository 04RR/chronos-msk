import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import json
import os
from transformers import SiglipVisionModel
from peft import PeftModel

# --- Custom Pooling ---
class GlobalAveragePooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.mean(dim=1))

class RegressorAgent:
    def __init__(self, model_dir, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🛡️ Regressor: Loading from {model_dir}...")

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self._build_model(model_dir)

    def _build_model(self, model_dir):
        base = SiglipVisionModel.from_pretrained(self.config["model_id"], torch_dtype=torch.float32)
        adapter_path = os.path.join(model_dir, "adapter")
        self.backbone = PeftModel.from_pretrained(base, adapter_path)

        hidden = self.config["hidden_size"]
        self.pooler = GlobalAveragePooling(hidden)
        self.gender_embed = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 128))
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden + 128),
            nn.Linear(hidden + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.config["num_bins"]),
        )

        heads_path = os.path.join(model_dir, "heads.pth")
        state = torch.load(heads_path, map_location=self.device)
        self.pooler.load_state_dict(state["pooler"])
        self.gender_embed.load_state_dict(state["gender_embed"])
        self.classifier.load_state_dict(state["classifier"])

        self.backbone.to(self.device).eval()
        self.pooler.to(self.device).eval()
        self.gender_embed.to(self.device).eval()
        self.classifier.to(self.device).eval()

    def _inference_single(self, img_bgr, is_male):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = self.config["image_size"]
        scale = s / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        res = cv2.resize(img, (nw, nh))
        pad = np.zeros((s, s, 3), dtype=np.uint8)
        top, left = (s - nh) // 2, (s - nw) // 2
        pad[top : top + nh, left : left + nw] = res

        norm = (pad.astype(np.float32) / 255.0 - 0.5) / 0.5
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t_sex = torch.tensor([[1.0 if is_male else 0.0]], device=self.device)

        with torch.no_grad():
            backbone_out = self.backbone(pixel_values=t_img)
            vis = self.pooler(backbone_out.last_hidden_state)
            gen = self.gender_embed(t_sex)
            logits = self.classifier(torch.cat([vis, gen], dim=-1))
            probs = F.softmax(logits, dim=-1)
            ages = torch.arange(probs.shape[-1], device=self.device)
            months = torch.sum(probs * ages).item()

        return months

    def predict(self, image_path, is_male):
        """
        Runs TTA. Returns average age in MONTHS.
        """
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Regressor could not read image: {image_path}")

        pred_1 = self._inference_single(original, is_male)
        pred_2 = self._inference_single(cv2.flip(original, 1), is_male)
        
        avg_months = (pred_1 + pred_2) / 2.0
        
        # --- CRITICAL FIX: RETURN MONTHS (NO / 12.0) ---
        return round(avg_months, 2)