"""
train_retriever_sota.py
========================
SOTA retrieval training directly from Digital Hand Atlas images.

Directory structure:
  JPEGimages/
    ASIF/ASIF00/5599.jpg   (Asian Female, age 0 years)
    ASIF/ASIF01/...         (Asian Female, age 1 year)
    ...
    ASIM/ASIM00/...         (Asian Male, age 0 years)
    BLKF/BLKF00/...         (Black Female, age 0 years)
    CAUM/CAUM15/...         (Caucasian Male, age 15 years)
    HISF/HISF10/...         (Hispanic Female, age 10 years)

Naming convention:
  {RACE_CODE}{GENDER_CODE}{AGE_YEARS:02d}/{image_id}.jpg
  RACE: ASI=Asian, BLK=Black, CAU=Caucasian, HIS=Hispanic
  GENDER: F=Female, M=Male

Training approach:
  1. Load raw images → MedSigLIP (frozen) → 1152-D embeddings
  2. Cache all embeddings (one-time cost)
  3. Train DemographicProjector: 1152-D → 256-D
  4. Multi-loss: MS Loss + Proxy-NCA + Soft Contrastive + Age Regression
  5. Rebuild FAISS indices with new projector
"""

import os
import re
import json
import math
import glob
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
from tqdm import tqdm
from transformers import SiglipVisionModel, AutoProcessor

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Data
    "atlas_dir": "/mnt/e/chronos-msk/data/Digital Hand Atlas/Digital Hand Atlas/JPEGimages",
    "embedding_cache": "weights/atlas_embeddings_cache.npz",
    "embed_model_id": "google/medsiglip-448",
    
    # Output
    "output_dir": "weights",
    "output_name": "projector_sota.pth",
    
    # Architecture
    "input_dim": 1152,
    "output_dim": 256,
    "hidden_dim": 768,
    
    # Training
    "batch_size": 256,
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 20,
    "min_lr": 1e-6,
    
    # Loss weights
    "ms_loss_weight": 1.0,
    "proxy_loss_weight": 1.0,
    "contrastive_weight": 0.5,
    "age_loss_weight": 0.5,
    
    # MS Loss params
    "ms_alpha": 2.0,
    "ms_beta": 50.0,
    "ms_base": 0.5,
    "ms_margin": 0.1,
    
    # Proxy-NCA
    "proxy_temperature": 0.1,
    
    # Curriculum
    "age_curriculum_start": 36.0,   # months (start easy)
    "age_curriculum_end": 6.0,      # months (end tight)
    "curriculum_warmup_epochs": 100,
    
    # Validation
    "val_fraction": 0.1,
    "early_stopping_patience": 50,
    
    # Race/Gender mapping
    "folder_to_partition": {
        "ASIF": ("Female", "Asian"),
        "ASIM": ("Male", "Asian"),
        "BLKF": ("Female", "Black"),
        "BLKM": ("Male", "Black"),
        "CAUF": ("Female", "Caucasian"),
        "CAUM": ("Male", "Caucasian"),
        "HISF": ("Female", "Hispanic"),
        "HISM": ("Male", "Hispanic"),
    },
}

# Build class list and mapping
PARTITIONS = [
    "Male_Asian", "Male_Caucasian", "Male_Hispanic", "Male_Black",
    "Female_Asian", "Female_Caucasian", "Female_Hispanic", "Female_Black",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(PARTITIONS)}


# ============================================================
# STEP 1: SCAN ATLAS DIRECTORY
# ============================================================
def scan_atlas(atlas_dir):
    """
    Scans the Digital Hand Atlas directory structure.
    Returns list of dicts: {path, gender, race, age_years, age_months, partition, class_idx}
    """
    entries = []
    folder_map = CONFIG["folder_to_partition"]
    
    print(f"📂 Scanning {atlas_dir}...")
    
    for race_gender_folder in sorted(os.listdir(atlas_dir)):
        folder_path = os.path.join(atlas_dir, race_gender_folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Map folder code to (gender, race)
        code = race_gender_folder.upper()
        if code not in folder_map:
            print(f"  ⚠️ Unknown folder: {race_gender_folder}")
            continue
        
        gender, race = folder_map[code]
        partition = f"{gender}_{race}"
        cls_idx = CLASS_TO_IDX.get(partition, -1)
        
        if cls_idx == -1:
            continue
        
        # Scan age subfolders: e.g., ASIF00, ASIF01, ..., ASIF18
        for age_folder in sorted(os.listdir(folder_path)):
            age_folder_path = os.path.join(folder_path, age_folder)
            if not os.path.isdir(age_folder_path):
                continue
            
            # Extract age from folder name: last 2 chars are age in years
            # e.g., ASIF00 → 00, ASIF15 → 15, CAUM08 → 08
            match = re.search(r'(\d{2})$', age_folder)
            if not match:
                print(f"  ⚠️ Cannot parse age from: {age_folder}")
                continue
            
            age_years = int(match.group(1))
            age_months = age_years * 12  # Convert to months
            
            # Scan images in this age folder
            for img_file in os.listdir(age_folder_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    continue
                
                img_path = os.path.join(age_folder_path, img_file)
                
                entries.append({
                    "path": img_path,
                    "gender": gender,
                    "race": race,
                    "age_years": age_years,
                    "age_months": age_months,
                    "partition": partition,
                    "class_idx": cls_idx,
                    "image_id": os.path.splitext(img_file)[0],
                })
    
    print(f"✅ Found {len(entries)} images")
    
    # Summary
    partition_counts = defaultdict(int)
    age_counts = defaultdict(int)
    for e in entries:
        partition_counts[e["partition"]] += 1
        age_counts[e["age_years"]] += 1
    
    print(f"\n  Partition distribution:")
    for p in sorted(partition_counts):
        print(f"    {p}: {partition_counts[p]}")
    
    print(f"\n  Age distribution (years):")
    for a in sorted(age_counts):
        print(f"    {a:2d}y ({a*12:3d}m): {age_counts[a]} images")
    
    return entries


# ============================================================
# STEP 2: EMBED ALL IMAGES WITH MedSigLIP (cached)
# ============================================================
def letterbox_resize(image, size=448):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    padded[top : top + nh, left : left + nw] = resized
    return padded


def embed_atlas(entries, cache_path, model_id):
    """
    Embed all atlas images with MedSigLIP. Cache to disk.
    Returns: (embeddings [N, 1152], metadata list)
    """
    if os.path.exists(cache_path):
        print(f"⚡ Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = json.loads(str(data["metadata"]))
        print(f"   {len(embeddings)} embeddings loaded")
        return embeddings, metadata
    
    print(f"🐢 Embedding {len(entries)} atlas images with MedSigLIP...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipVisionModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    
    embeddings = []
    valid_metadata = []
    failed = 0
    
    for entry in tqdm(entries, desc="Embedding"):
        try:
            # Load image
            img_bgr = cv2.imread(entry["path"])
            if img_bgr is None:
                # Try PIL fallback for unusual formats
                try:
                    pil_img = Image.open(entry["path"]).convert("RGB")
                    img_bgr = np.array(pil_img)[:, :, ::-1]
                except:
                    failed += 1
                    continue
            
            # Preprocess: same as build_full_index.py
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = letterbox_resize(img_rgb, 448)
            
            # Embed
            inputs = processor(images=img_resized, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = model(**inputs).pooler_output
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(feat.cpu().numpy()[0])
            valid_metadata.append({
                "path": entry["path"],
                "gender": entry["gender"],
                "race": entry["race"],
                "age_months": entry["age_months"],
                "age_years": entry["age_years"],
                "partition": entry["partition"],
                "class_idx": entry["class_idx"],
                "image_id": entry["image_id"],
            })
            
        except Exception as e:
            failed += 1
            continue
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    print(f"✅ Embedded {len(embeddings)} images ({failed} failed)")
    
    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        embeddings=embeddings,
        metadata=json.dumps(valid_metadata),
    )
    print(f"💾 Cached to {cache_path}")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return embeddings, valid_metadata


# ============================================================
# DATASET
# ============================================================
class AtlasEmbeddingDataset(Dataset):
    def __init__(self, embeddings, metadata, indices):
        self.embeddings = embeddings[indices]
        self.labels = np.array([metadata[i]["class_idx"] for i in indices], dtype=np.int64)
        self.ages = np.array([metadata[i]["age_months"] for i in indices], dtype=np.float32)
        self.partitions = [metadata[i]["partition"] for i in indices]
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.tensor(self.labels[idx]),
            torch.tensor(self.ages[idx]),
        )
    
    def get_sample_weights(self):
        """Upweight underrepresented age bins."""
        age_bins = (self.ages // 12).astype(int)
        bin_counts = np.bincount(age_bins, minlength=20)
        bin_counts = np.maximum(bin_counts, 1)
        weights = np.array([1.0 / bin_counts[b] for b in age_bins])
        return weights / weights.sum() * len(weights)


# ============================================================
# MODEL
# ============================================================
class DemographicProjectorSOTA(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=768, num_classes=8):
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
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.proxies = nn.Parameter(torch.randn(num_classes, output_dim) * 0.1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_age=False):
        h = self.trunk(x) + self.skip(x)
        emb = self.projector(h)
        emb = F.normalize(emb, p=2, dim=1)
        if return_age:
            return emb, self.age_head(h).squeeze(-1)
        return emb
    
    def get_proxies(self):
        return F.normalize(self.proxies, p=2, dim=1)


# ============================================================
# LOSSES
# ============================================================
class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5, margin=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin
    
    def forward(self, embeddings, labels, ages, pos_thresh, neg_thresh):
        B = embeddings.size(0)
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        sim = torch.mm(embeddings, embeddings.t())
        age_diff = torch.abs(ages.unsqueeze(0) - ages.unsqueeze(1))
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        pos_mask = same_class & (age_diff <= pos_thresh)
        neg_mask = (~same_class) | (same_class & (age_diff >= neg_thresh))
        
        diag = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~diag
        neg_mask = neg_mask & ~diag
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        
        for i in range(B):
            pos_idx = torch.where(pos_mask[i])[0]
            neg_idx = torch.where(neg_mask[i])[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            
            pos_sim = sim[i, pos_idx]
            neg_sim = sim[i, neg_idx]
            
            neg_max = neg_sim.max().detach()
            pos_min = pos_sim.min().detach()
            
            p_keep = pos_sim < (neg_max + self.margin)
            n_keep = neg_sim > (pos_min - self.margin)
            
            pf = pos_sim[p_keep] if p_keep.sum() > 0 else pos_sim
            nf = neg_sim[n_keep] if n_keep.sum() > 0 else neg_sim
            
            pos_exp = torch.clamp(-self.alpha * (pf - self.base), max=50.0)
            neg_exp = torch.clamp(self.beta * (nf - self.base), max=50.0)
            
            loss += (1.0 / self.alpha) * torch.log(1 + torch.sum(torch.exp(pos_exp)))
            loss += (1.0 / self.beta) * torch.log(1 + torch.sum(torch.exp(neg_exp)))
            count += 1
        
        return loss / max(count, 1)


class ProxyNCALoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels, proxies):
        dist = torch.cdist(embeddings, proxies, p=2)
        logits = -dist / self.temperature
        return F.cross_entropy(logits, labels)


class SoftContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, sigma=12.0):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
    
    def forward(self, embeddings, ages, labels):
        B = embeddings.size(0)
        if B < 4:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        sim = torch.mm(embeddings, embeddings.t()) / self.temperature
        age_diff = torch.abs(ages.unsqueeze(0) - ages.unsqueeze(1))
        same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        soft_target = torch.exp(-age_diff / self.sigma) * same_class
        diag = torch.eye(B, device=embeddings.device)
        soft_target = soft_target * (1.0 - diag)
        
        row_sum = soft_target.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft_target = soft_target / row_sum
        
        sim_masked = sim - diag * 1e9
        log_prob = F.log_softmax(sim_masked, dim=1)
        
        valid = (row_sum.squeeze() > 1e-6)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return -(soft_target[valid] * log_prob[valid]).sum(dim=1).mean()


# ============================================================
# SCHEDULER & CURRICULUM
# ============================================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup:
            factor = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, blr * factor)


def get_curriculum_threshold(epoch):
    warmup = CONFIG["curriculum_warmup_epochs"]
    start = CONFIG["age_curriculum_start"]
    end = CONFIG["age_curriculum_end"]
    if epoch >= warmup:
        return end
    progress = epoch / warmup
    factor = 0.5 * (1 + math.cos(math.pi * progress))
    return end + (start - end) * factor


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, val_loader, device):
    model.eval()
    all_emb, all_lbl, all_age = [], [], []
    
    with torch.no_grad():
        for vecs, labels, ages in val_loader:
            emb = model(vecs.to(device))
            all_emb.append(emb.cpu())
            all_lbl.append(labels)
            all_age.append(ages)
    
    embs = torch.cat(all_emb).numpy()
    lbls = torch.cat(all_lbl).numpy()
    ages = torch.cat(all_age).numpy()
    
    n = len(embs)
    if n < 10:
        return {"recall@1": 0, "recall@5": 0, "age_mae": 999}
    
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs.astype(np.float32))
    
    k = min(6, n)
    D, I = index.search(embs.astype(np.float32), k)
    
    r1, r5, age_errs = 0, 0, []
    
    for i in range(n):
        nbrs = I[i, 1:k]
        if len(nbrs) == 0:
            continue
        n_ages = ages[nbrs]
        n_lbls = lbls[nbrs]
        correct = (n_lbls == lbls[i]) & (np.abs(n_ages - ages[i]) < 12)
        if len(correct) > 0 and correct[0]:
            r1 += 1
        if correct.any():
            r5 += 1
        age_errs.append(abs(n_ages[0] - ages[i]))
    
    model.train()
    return {
        "recall@1": r1 / n,
        "recall@5": r5 / n,
        "age_mae": float(np.mean(age_errs)) if age_errs else 999,
    }


# ============================================================
# REBUILD INDICES
# ============================================================
def rebuild_indices(model, embeddings, metadata, device):
    """
    Rebuild FAISS indices using the trained projector.
    Groups by partition, projects, and saves.
    """
    output_dir = "indices_projected_256d"
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Group by partition
    partition_data = defaultdict(lambda: {"vecs": [], "meta": []})
    
    for i, m in enumerate(metadata):
        partition = m["partition"]
        partition_data[partition]["vecs"].append(embeddings[i])
        partition_data[partition]["meta"].append({
            "id": m.get("image_id", str(i)),
            "age_months": float(m["age_months"]),
            "age_years": float(m["age_years"]),
            "gender": m["gender"],
            "race": m["race"],
            "path": m["path"],
        })
    
    print(f"\n📦 Rebuilding {len(partition_data)} partition indices...")
    
    for partition, data in partition_data.items():
        raw_vecs = np.array(data["vecs"], dtype=np.float32)
        
        # Project through trained model
        with torch.no_grad():
            tensor = torch.from_numpy(raw_vecs).to(device)
            projected = []
            for i in range(0, len(tensor), 256):
                chunk = tensor[i:i+256]
                proj = model(chunk).cpu().numpy()
                projected.append(proj)
            proj_vecs = np.vstack(projected).astype(np.float32)
        
        # Build FAISS index
        index = faiss.IndexFlatL2(CONFIG["output_dim"])
        index.add(proj_vecs)
        
        # Save
        idx_path = os.path.join(output_dir, f"{partition}.index")
        meta_path = os.path.join(output_dir, f"{partition}_meta.json")
        
        faiss.write_index(index, idx_path)
        with open(meta_path, "w") as f:
            json.dump(data["meta"], f, indent=2)
        
        print(f"  {partition}: {index.ntotal} vectors, "
              f"ages {min(m['age_months'] for m in data['meta']):.0f}-"
              f"{max(m['age_months'] for m in data['meta']):.0f}m")
    
    print(f"✅ All indices saved to {output_dir}")


# ============================================================
# MAIN TRAINING
# ============================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Step 1: Scan atlas
    entries = scan_atlas(CONFIG["atlas_dir"])
    if not entries:
        print("❌ No images found!")
        return
    
    # Step 2: Embed (or load cache)
    embeddings, metadata = embed_atlas(
        entries, CONFIG["embedding_cache"], CONFIG["embed_model_id"]
    )
    
    # Step 3: Train/val split
    n = len(embeddings)
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    split = int(n * (1 - CONFIG["val_fraction"]))
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    print(f"\n📊 Split: {len(train_idx)} train, {len(val_idx)} val")
    
    train_ds = AtlasEmbeddingDataset(embeddings, metadata, train_idx)
    val_ds = AtlasEmbeddingDataset(embeddings, metadata, val_idx)
    
    # Balanced sampling
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    
    effective_batch = min(CONFIG["batch_size"], len(train_ds))
    
    train_loader = DataLoader(
        train_ds, batch_size=effective_batch, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(val_ds), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    # Model
    model = DemographicProjectorSOTA(
        CONFIG["input_dim"], CONFIG["output_dim"],
        CONFIG["hidden_dim"], len(PARTITIONS),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineWarmupScheduler(
        optimizer, CONFIG["warmup_epochs"], CONFIG["epochs"], CONFIG["min_lr"],
    )
    
    # Losses
    ms_loss_fn = MultiSimilarityLoss(
        CONFIG["ms_alpha"], CONFIG["ms_beta"],
        CONFIG["ms_base"], CONFIG["ms_margin"],
    )
    proxy_loss_fn = ProxyNCALoss(CONFIG["proxy_temperature"])
    soft_contrast_fn = SoftContrastiveLoss(temperature=0.1, sigma=12.0)
    age_loss_fn = nn.SmoothL1Loss()
    
    # State
    best_recall = -1
    best_age_mae = 999
    patience = 0
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_name"])
    training_log = []
    
    print(f"\n🚀 Training ({CONFIG['epochs']} epochs, batch={effective_batch})")
    print("=" * 80)
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        
        pos_thresh = get_curriculum_threshold(epoch)
        neg_thresh = pos_thresh * 3
        
        epoch_loss = {"ms": 0, "proxy": 0, "sc": 0, "age": 0, "total": 0}
        n_batches = 0
        
        for batch_vecs, batch_labels, batch_ages in train_loader:
            batch_vecs = batch_vecs.to(device)
            batch_labels = batch_labels.to(device)
            batch_ages = batch_ages.to(device)
            
            embeddings_out, age_pred = model(batch_vecs, return_age=True)
            proxies = model.get_proxies()
            
            loss_ms = ms_loss_fn(embeddings_out, batch_labels, batch_ages, pos_thresh, neg_thresh)
            loss_proxy = proxy_loss_fn(embeddings_out, batch_labels, proxies)
            loss_sc = soft_contrast_fn(embeddings_out, batch_ages, batch_labels)
            loss_age = age_loss_fn(age_pred, batch_ages)
            
            total = (
                CONFIG["ms_loss_weight"] * loss_ms
                + CONFIG["proxy_loss_weight"] * loss_proxy
                + CONFIG["contrastive_weight"] * loss_sc
                + CONFIG["age_loss_weight"] * loss_age
            )
            
            # Debug first batch of first epoch
            if epoch == 0 and n_batches == 0:
                with torch.no_grad():
                    age_diff = torch.abs(batch_ages.unsqueeze(0) - batch_ages.unsqueeze(1))
                    same_cls = batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)
                    n_pos = (same_cls & (age_diff <= pos_thresh)).sum().item()
                    n_neg = ((~same_cls) | (same_cls & (age_diff >= neg_thresh))).sum().item()
                    B = len(batch_vecs)
                
                print(f"\n  🔍 FIRST BATCH DEBUG:")
                print(f"     Batch size:     {B}")
                print(f"     Unique classes: {len(batch_labels.unique())}")
                print(f"     Age range:      {batch_ages.min():.0f}-{batch_ages.max():.0f}m")
                print(f"     Pos threshold:  {pos_thresh:.1f}m")
                print(f"     Neg threshold:  {neg_thresh:.1f}m")
                print(f"     Positive pairs: {n_pos} ({100*n_pos/(B*B):.1f}%)")
                print(f"     Negative pairs: {n_neg} ({100*n_neg/(B*B):.1f}%)")
                print(f"     Loss MS:        {loss_ms.item():.6f}")
                print(f"     Loss Proxy:     {loss_proxy.item():.6f}")
                print(f"     Loss SC:        {loss_sc.item():.6f}")
                print(f"     Loss Age:       {loss_age.item():.4f}")
                print(f"     Total:          {total.item():.4f}")
                print(f"     Emb std:        {embeddings_out.std().item():.6f}")
                print()
            
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss["ms"] += loss_ms.item()
            epoch_loss["proxy"] += loss_proxy.item()
            epoch_loss["sc"] += loss_sc.item()
            epoch_loss["age"] += loss_age.item()
            epoch_loss["total"] += total.item()
            n_batches += 1
        
        for k in epoch_loss:
            epoch_loss[k] /= max(n_batches, 1)
        
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]["lr"]
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Log
        log_entry = {"epoch": epoch + 1, "lr": float(lr), "pos_thresh": float(pos_thresh)}
        for k, v in epoch_loss.items():
            log_entry[f"loss_{k}"] = float(v)
        for k, v in val_metrics.items():
            log_entry[f"val_{k}"] = float(v)
        training_log.append(log_entry)
        
        # Print
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  E{epoch+1:3d}/{CONFIG['epochs']} | "
                f"LR:{lr:.1e} | "
                f"L:{epoch_loss['total']:.4f} "
                f"(MS:{epoch_loss['ms']:.3f} "
                f"Px:{epoch_loss['proxy']:.3f} "
                f"SC:{epoch_loss['sc']:.3f} "
                f"Ag:{epoch_loss['age']:.1f}) | "
                f"R@1:{val_metrics['recall@1']:.3f} "
                f"R@5:{val_metrics['recall@5']:.3f} "
                f"AgMAE:{val_metrics['age_mae']:.1f}m | "
                f"PT:{pos_thresh:.1f}m"
            )
        
        # Save best
        improved = False
        if val_metrics["recall@1"] > best_recall + 0.003:
            best_recall = val_metrics["recall@1"]
            improved = True
        elif (abs(val_metrics["recall@1"] - best_recall) < 0.01
              and val_metrics["age_mae"] < best_age_mae - 0.3):
            improved = True
        
        if improved:
            best_age_mae = min(best_age_mae, val_metrics["age_mae"])
            patience = 0
            torch.save(model.state_dict(), output_path)
            print(f"  💾 SAVED (R@1={val_metrics['recall@1']:.3f}, "
                  f"AgMAE={val_metrics['age_mae']:.1f}m)")
        else:
            patience += 1
            if patience >= CONFIG["early_stopping_patience"]:
                print(f"\n  ⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Save log
    log_path = os.path.join(CONFIG["output_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    
    # Rebuild indices with best model
    print(f"\n{'='*70}")
    print(f"  📦 Rebuilding FAISS indices with trained projector...")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    rebuild_indices(model, embeddings, metadata, device)
    
    print(f"\n{'='*70}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Best R@1:     {best_recall:.4f}")
    print(f"  Best Age MAE: {best_age_mae:.1f}m")
    print(f"  Weights:      {output_path}")
    print(f"  Indices:      indices_projected_256d/")
    print(f"\n  NEXT STEPS:")
    print(f"  1. rm -rf evaluation_results_embedding_fix/")
    print(f"  2. python eval_embedding_fix.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("  CHRONOS-MSK SOTA RETRIEVAL TRAINING")
    print("  Direct from Digital Hand Atlas images")
    print("🔬" * 35 + "\n")
    train()