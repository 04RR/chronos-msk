import faiss
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DemographicProjectorSOTA(nn.Module):
    """SOTA projector — must match train_retriever_sota.py architecture."""
    def __init__(self, input_dim=1152, output_dim=256, hidden_dim=768, num_classes=8):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.proxies = nn.Parameter(torch.randn(num_classes, output_dim) * 0.01)

    def forward(self, x, return_age=False):
        h = self.trunk(x) + self.skip(x)
        emb = self.projector(h)
        emb = F.normalize(emb, p=2, dim=1)
        if return_age:
            return emb, self.age_head(h).squeeze(-1)
        return emb


# Legacy projector for backward compatibility
class DemographicProjector(nn.Module):
    def __init__(self, input_dim=1152, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class ArchivistAgent:
    def __init__(self, indices_dir, projector_path=None, dha_csv_path=None):
        print(f"📚 Archivist: Loading from {indices_dir}...")
        self.indices_dir = indices_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.projector = None
        if projector_path and os.path.exists(projector_path):
            print(f"   Loading Projector from {projector_path}")
            self.projector = self._load_projector(projector_path)
        
        self.partition_meta = {}
        meta_count = 0
        for f in os.listdir(indices_dir):
            if f.endswith("_meta.json"):
                key = f.replace("_meta.json", "")
                try:
                    with open(os.path.join(indices_dir, f)) as fh:
                        self.partition_meta[key] = json.load(fh)
                    meta_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to load {f}: {e}")
        print(f"   Loaded metadata for {meta_count} partitions")
    
    def _load_projector(self, path):
        """Load projector, auto-detecting architecture."""
        state_dict = torch.load(path, map_location=self.device)
        
        # Detect SOTA vs Legacy by checking for 'trunk' keys
        has_trunk = any(k.startswith("trunk.") for k in state_dict.keys())
        
        if has_trunk:
            print(f"   Detected SOTA projector architecture")
            model = DemographicProjectorSOTA(1152, 256, 768, 8).to(self.device)
        else:
            print(f"   Detected legacy projector architecture")
            model = DemographicProjector(1152, 256).to(self.device)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    
    def _project_embedding(self, raw_embedding):
        if self.projector is None:
            return raw_embedding.astype("float32").reshape(1, -1)
        with torch.no_grad():
            tensor = torch.tensor(raw_embedding).float().to(self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            proj = self.projector(tensor).cpu().numpy()
            return proj.astype("float32")

    def retrieve(self, query_embedding, sex, race, top_k=3):
        q_emb = self._project_embedding(query_embedding)
        target_key = f"{sex}_{race}"
        index_path = os.path.join(self.indices_dir, f"{target_key}.index")
        if not os.path.exists(index_path):
            return []
        index = faiss.read_index(index_path)
        D, I = index.search(q_emb, top_k)
        meta_list = self.partition_meta.get(target_key, [])
        results = []
        for rank in range(len(I[0])):
            idx = int(I[0][rank])
            if idx == -1:
                continue
            match_data = {
                "internal_id": idx,
                "distance": float(D[0][rank]),
                "partition": target_key,
                "age_months": -1.0,
            }
            if idx < len(meta_list):
                match_data["age_months"] = float(
                    meta_list[idx].get("age_months", -1.0)
                )
            results.append(match_data)
        return results
