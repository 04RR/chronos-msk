"""
SOTA MedSigLIP Bone Age - RTX 4090 OPTIMIZED
=============================================

Optimizations for RTX 4090 (Ada Lovelace):
1. Gradient checkpointing for memory efficiency
2. Flash Attention via SDPA
3. TF32 for faster matmuls
4. Larger batch sizes
5. Optimized data loading
6. Proper CUDA warmup
7. Mixed precision via Trainer (bf16)
"""

import os
import json
import time
import warnings
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

# Optimize OpenCV
cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)

from sklearn.model_selection import train_test_split
from transformers import (
    SiglipVisionModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from peft import get_peft_model, LoraConfig

# Albumentations
try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ==========================================
# 1. RTX 4090 OPTIMIZED CONFIGURATION
# ==========================================
@dataclass
class Config:
    """RTX 4090 FULLY optimized configuration."""

    # Paths
    csv_path: str = "data/RSNA/boneage-dataset.csv"
    image_folder: str = "data/RSNA/boneage-training-dataset/boneage-training-dataset"
    model_id: str = "google/medsiglip-448"
    output_dir: str = "./medsiglip_bone_age_sota"

    # Image
    image_size: int = 448
    image_mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, ...] = (0.5, 0.5, 0.5)

    # =============================================
    # OPTIMIZED FOR RTX 4090 - MAXIMIZE GPU USAGE
    # =============================================
    batch_size: int = 8  # ⬆️ Increased from 8 (you have memory!)
    gradient_accumulation_steps: int = 2  # ⬇️ No need to accumulate
    gradient_checkpointing: bool = True  # ⬇️ Disabled for speed

    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # DoRA
    use_dora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # DLDL
    num_age_bins: int = 228
    sigma: float = 12.0
    label_smoothing: float = 0.01

    # Pooling
    pooling_type: str = "gap"

    # Augmentation
    use_augmentation: bool = True
    rotation_limit: int = 20

    # torch.compile - can provide 10-20% speedup
    use_compile: bool = False  # Try True if no errors
    compile_mode: str = "reduce-overhead"

    # Data loading
    num_workers: int = 2
    pin_memory: bool = False
    prefetch_factor: int = 4
    persistent_workers: bool = True
    cache_images: bool = True

    # Mixed precision
    use_bf16: bool = True

    seed: int = 42


# ==========================================
# 2. CUDA OPTIMIZATION SETUP
# ==========================================
def setup_cuda_optimizations(config: Config):
    """Configure CUDA for maximum RTX 4090 performance."""

    print("\n" + "=" * 60)
    print("  🚀 RTX 4090 CUDA OPTIMIZATIONS")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ❌ CUDA not available!")
        return

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {gpu_name}")
    print(f"  Memory: {gpu_mem:.1f} GB")

    # 1. Enable TF32 for faster matmuls (Ada Lovelace)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("  ✓ TF32 enabled (faster matmuls)")

    # 2. cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("  ✓ cuDNN benchmark mode (auto-tuning)")

    # 3. Memory optimizations
    torch.cuda.empty_cache()

    # 4. Set memory allocation strategy
    if hasattr(torch.cuda, "memory"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 5. Enable Flash Attention if available (PyTorch 2.0+)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("  ✓ Flash Attention available (SDPA)")

    # 6. Gradient checkpointing status
    if config.gradient_checkpointing:
        print("  ✓ Gradient checkpointing enabled (memory saving mode)")
    else:
        print("  ✓ Gradient checkpointing disabled (speed mode)")

    print("=" * 60 + "\n")


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# 3. NUMERICALLY STABLE DLDL
# ==========================================
class DLDLUtils:
    """Numerically stable DLDL utilities."""

    @staticmethod
    def create_gaussian_distribution(
        age_months: float,
        num_bins: int = 228,
        sigma: float = 12.0,
        smoothing: float = 0.01,
    ) -> torch.Tensor:
        """Create smoothed Gaussian distribution."""
        ages = torch.arange(0, num_bins, dtype=torch.float32)
        dist = torch.exp(-((ages - age_months) ** 2) / (2 * sigma**2))
        dist = dist / (dist.sum() + 1e-8)

        # Label smoothing prevents zero probabilities
        smoothed = (1.0 - smoothing) * dist + smoothing / num_bins
        return smoothed / smoothed.sum()

    @staticmethod
    def expectation_regression(probs: torch.Tensor) -> torch.Tensor:
        """Calculate expected age from probability distribution."""
        device = probs.device
        ages = torch.arange(probs.shape[-1], dtype=torch.float32, device=device)
        if probs.dim() == 1:
            return torch.sum(probs * ages)
        return torch.sum(probs * ages, dim=-1)

    @staticmethod
    def stable_kl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Numerically stable KL divergence loss."""
        log_probs = F.log_softmax(logits, dim=-1)
        target = torch.clamp(target, min=1e-8)
        return F.kl_div(log_probs, target, reduction="batchmean")


# ==========================================
# 4. FAST IMAGE PROCESSING
# ==========================================
def letterbox_resize(image: np.ndarray, size: int = 448) -> np.ndarray:
    """Optimized letterbox resize maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized

    return padded


def normalize_image(
    image: np.ndarray, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
) -> torch.Tensor:
    """Fast normalization using numpy operations."""
    img = image.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return torch.from_numpy(img.transpose(2, 0, 1))


# ==========================================
# 5. AUGMENTATION
# ==========================================
class FastAugmentation:
    """Fast, compatible augmentation using albumentations."""

    def __init__(self, config: Config):
        self.config = config

        if ALBUMENTATIONS_AVAILABLE and config.use_augmentation:
            self.transform = A.Compose(
                [
                    A.Rotate(
                        limit=config.rotation_limit,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                ]
            )
        else:
            self.transform = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.transform:
            return self.transform(image=image)["image"]
        return image


# ==========================================
# 6. OPTIMIZED DATASET
# ==========================================
class BoneAgeDataset(Dataset):
    """Dataset with optional memory caching for slow filesystems."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str,
        is_train: bool,
        config: Config,
        cache_images: bool = True,
    ):  # Add caching option
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.is_train = is_train
        self.config = config
        self.aug = FastAugmentation(config) if is_train else None
        self.cache_images = cache_images

        # Pre-compute labels
        self.labels = []
        for idx in range(len(self.df)):
            age = min(float(self.df.iloc[idx]["boneage"]), config.num_age_bins - 1)
            self.labels.append(
                DLDLUtils.create_gaussian_distribution(
                    age, config.num_age_bins, config.sigma, config.label_smoothing
                )
            )

        # Cache images in memory (for slow filesystems like WSL2 mounts)
        self.image_cache = {}
        if cache_images:
            print(f"  Caching {len(self.df)} images in memory...")
            import tqdm

            for idx in tqdm.tqdm(range(len(self.df)), desc="  Loading images"):
                row = self.df.iloc[idx]
                img_path = os.path.join(self.root_dir, f"{row['id']}.png")
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Resize and cache (reduces memory by ~4x vs caching full size)
                    image = letterbox_resize(image, config.image_size)
                    self.image_cache[idx] = image
            print(f"  ✓ Cached {len(self.image_cache)} images")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load from cache or disk
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx].copy()  # Copy to avoid modifying cache
        else:
            img_path = os.path.join(self.root_dir, f"{row['id']}.png")
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = letterbox_resize(image, self.config.image_size)

        # Augment (only for training)
        if self.aug:
            image = self.aug(image)

        pixels = normalize_image(image, self.config.image_mean, self.config.image_std)

        return {
            "pixel_values": pixels,
            "gender": torch.tensor(
                1.0 if row.get("male", True) else 0.0, dtype=torch.float32
            ),
            "labels": self.labels[idx],
            "age_months": torch.tensor(float(row["boneage"]), dtype=torch.float32),
        }


# ==========================================
# 7. POOLING LAYERS
# ==========================================
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling with LayerNorm."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.mean(dim=1))


class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable query."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.0
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.norm(out.squeeze(1))


# ==========================================
# 8. MODEL WITH OPTIONAL GRADIENT CHECKPOINTING
# ==========================================
class BoneAgeModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        print(f"\n  Loading: {config.model_id}")

        # Load backbone
        self.backbone = SiglipVisionModel.from_pretrained(
            config.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.hidden_size = self.backbone.config.hidden_size
        print(f"  Hidden size: {self.hidden_size}")

        # Apply DoRA FIRST (before gradient checkpointing)
        if config.use_dora:
            self._apply_dora()

        # Enable gradient checkpointing AFTER PEFT wrapping
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("  ✓ Gradient checkpointing enabled (saves ~50% VRAM)")

        # Pooling layer
        if config.pooling_type == "attention":
            self.pooler = AttentionPooling(self.hidden_size)
        else:
            self.pooler = GlobalAveragePooling(self.hidden_size)

        # Gender embedding
        self.gender_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size + 128),
            nn.Linear(self.hidden_size + 128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, config.num_age_bins),
        )

        self._init_weights()

        self.pooler = self.pooler.to(torch.bfloat16)
        self.gender_embed = self.gender_embed.to(torch.bfloat16)
        self.classifier = self.classifier.to(torch.bfloat16)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Forward the checkpointing enable call to the inner backbone."""
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def _apply_dora(self):
        """Apply DoRA (Weight-Decomposed Low-Rank Adaptation) adapters."""
        target_modules = []
        for name, mod in self.backbone.named_modules():
            if isinstance(mod, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]:
                    target_modules.append(leaf)

        target_modules = list(set(target_modules))
        print(f"  DoRA targets: {target_modules}")

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_dora=True,
        )

        self.backbone = get_peft_model(self.backbone, lora_config)

        # Re-enable gradient checkpointing after PEFT wrapping
        if self.config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.backbone.print_trainable_parameters()

    def _init_weights(self):
        """Initialize custom head weights."""
        for mod in [self.classifier, self.gender_embed]:
            for m in mod:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        gender: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        age_months: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        # Visual features from backbone
        out = self.backbone(pixel_values=pixel_values)
        visual = self.pooler(out.last_hidden_state)

        # Gender embedding - ensure same dtype as visual features
        gender_feat = self.gender_embed(gender.unsqueeze(-1).to(visual.dtype))

        # Combine and classify
        combined = torch.cat([visual, gender_feat], dim=-1)
        logits = self.classifier(combined)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = DLDLUtils.stable_kl_loss(logits, labels.to(logits.dtype))

        return {"loss": loss, "logits": logits, "age_months": age_months}


# ==========================================
# 9. DATA COLLATOR
# ==========================================
class DataCollator:
    """Collate function for batching samples."""

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "gender": torch.stack([f["gender"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "age_months": torch.stack([f["age_months"] for f in features]),
        }


# ==========================================
# 10. CUSTOM TRAINER WITH TIMING
# ==========================================
class BoneAgeTrainer(Trainer):
    """Custom trainer with timing diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_times = []
        self.warmup_done = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss from model outputs."""
        outputs = model(
            pixel_values=inputs["pixel_values"],
            gender=inputs["gender"],
            labels=inputs["labels"],
            age_months=inputs["age_months"],
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with timing - updated signature for transformers >= 4.46."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Call parent training_step with new signature
        loss = super().training_step(model, inputs, num_items_in_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        self.step_times.append(elapsed)

        # Print timing for first 5 steps
        n = len(self.step_times)
        if n <= 5:
            mem_used = (
                torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available()
                else 0
            )
            print(f"\n  ⏱️  Step {n}: {elapsed:.2f}s | GPU Memory: {mem_used:.1f}GB")
        elif n == 6 and not self.warmup_done:
            avg = sum(self.step_times[1:]) / (len(self.step_times) - 1)
            print(f"\n  📊 Avg step time (after warmup): {avg:.2f}s")
            steps_per_epoch = (
                len(self.train_dataset) // self.args.per_device_train_batch_size
            )
            print(f"     Expected epoch time: ~{avg * steps_per_epoch / 60:.1f} min")
            self.warmup_done = True

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Prediction step for evaluation."""
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                gender=inputs["gender"],
                labels=inputs["labels"],
                age_months=inputs["age_months"],
            )

        if prediction_loss_only:
            return (outputs["loss"], None, None)
        return (outputs["loss"], outputs["logits"], inputs["age_months"])


# ==========================================
# 11. METRICS
# ==========================================
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits, dim=-1)
    pred = DLDLUtils.expectation_regression(probs)
    true = torch.tensor(labels, dtype=torch.float32)

    errors = torch.abs(pred - true)

    return {
        "mae_months": errors.mean().item(),
        "mae_years": errors.mean().item() / 12,
        "rmse_months": torch.sqrt((errors**2).mean()).item(),
        "p90_months": torch.quantile(errors, 0.9).item(),
    }


# ==========================================
# 12. BENCHMARK FUNCTION
# ==========================================
def benchmark_model(model: nn.Module, config: Config, device: str = "cuda"):
    """Benchmark with realistic batch size."""
    print("\n" + "=" * 60)
    print("  🔥 RTX 4090 BENCHMARK")
    print("=" * 60)

    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Use batch_size=2 for benchmark to avoid OOM
    B = 2  # Fixed small batch for benchmark
    dummy = {
        "pixel_values": torch.randn(
            B,
            3,
            config.image_size,
            config.image_size,
            dtype=torch.bfloat16,
        ).to(device),
        "gender": torch.ones(B, dtype=torch.bfloat16).to(device),
        "labels": torch.randn(B, config.num_age_bins, dtype=torch.bfloat16)
        .softmax(dim=-1)
        .to(device),
    }

    # Warmup
    print(f"  Warming up with batch_size={B}...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy["pixel_values"], dummy["gender"], dummy["labels"])
        torch.cuda.synchronize()

    # Benchmark forward
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy["pixel_values"], dummy["gender"], dummy["labels"])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    fwd_time = sum(times) / len(times)

    # Clear before backward benchmark
    torch.cuda.empty_cache()

    # Benchmark forward + backward
    model.train()
    times_fb = []

    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()

        out = model(dummy["pixel_values"], dummy["gender"], dummy["labels"])
        loss = out["loss"]
        loss.backward()

        torch.cuda.synchronize()
        times_fb.append(time.perf_counter() - start)
        model.zero_grad(set_to_none=True)  # More efficient
        torch.cuda.empty_cache()

    fb_time = sum(times_fb) / len(times_fb)
    mem_used = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n  📊 BENCHMARK RESULTS (batch_size={B}):")
    print(f"     Forward pass:     {fwd_time*1000:.0f}ms")
    print(f"     Forward+Backward: {fb_time*1000:.0f}ms")
    print(f"     GPU Memory:       {mem_used:.1f}GB / 24GB")

    # Check if memory is reasonable
    if mem_used > 20:
        print(f"  ⚠️  WARNING: High memory usage! Consider smaller batch/rank")

    # Extrapolate to actual batch size
    actual_batch = config.batch_size
    estimated_throughput = actual_batch / (fb_time * actual_batch / B)
    print(
        f"     Est. throughput (batch={actual_batch}): ~{estimated_throughput:.0f} samples/sec"
    )

    print("=" * 60 + "\n")

    torch.cuda.empty_cache()
    return fb_time


# ==========================================
# 13. MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("\n" + "=" * 70)
    print("  🚀 BONE AGE ASSESSMENT - RTX 4090 OPTIMIZED")
    print("=" * 70)

    config = Config()
    set_seed(config.seed)

    # Setup CUDA optimizations
    setup_cuda_optimizations(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print("[1/6] Loading data...")
    df = pd.read_csv(config.csv_path)
    if "male" not in df.columns:
        df["male"] = True
    df["boneage"] = df["boneage"].clip(0, config.num_age_bins - 1)

    print(f"  Samples: {len(df):,}")
    print(f"  Age range: {df['boneage'].min():.0f} - {df['boneage'].max():.0f} months")

    # Split data
    print("\n[2/6] Splitting data...")
    df["age_bin"] = pd.cut(df["boneage"], bins=10, labels=False)
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=config.seed, stratify=df["age_bin"]
    )
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,}")

    # Create datasets
    print("\n[3/6] Creating datasets...")
    train_ds = BoneAgeDataset(train_df, config.image_folder, True, config)
    val_ds = BoneAgeDataset(val_df, config.image_folder, False, config)

    # Create model
    print("\n[4/6] Creating model...")
    model = BoneAgeModel(config)
    model.to(device)

    # Benchmark
    benchmark_model(model, config, device)

    # Training arguments
    print("[5/6] Configuring training...")
    os.makedirs(config.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        # Epochs & batching - FIXED
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,  # Now 2
        per_device_eval_batch_size=config.batch_size,  # Same for eval
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # Now 4
        # Optimizer
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=config.max_grad_norm,
        optim="adamw_torch_fused",
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        # Data loading - REDUCED for WSL2
        dataloader_num_workers=4,  # ⬇️ WSL2 has overhead with many workers
        dataloader_pin_memory=config.pin_memory,
        dataloader_prefetch_factor=2,  # ⬇️ Reduced
        dataloader_persistent_workers=True,
        # Mixed precision
        bf16=True,
        bf16_full_eval=True,
        # Gradient checkpointing - ENABLED
        gradient_checkpointing=config.gradient_checkpointing,  # True
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Other
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="mae_months",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        # Memory optimization
        torch_empty_cache_steps=100,  # Clear cache periodically
    )

    # Create trainer
    trainer = BoneAgeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollator(),
    )

    # Print configuration summary
    print("\n" + "=" * 70)
    print("  CONFIGURATION SUMMARY")
    print("=" * 70)
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    print(
        f"""
  Model: {config.model_id}
  Pooling: {config.pooling_type}
  DoRA: r={config.lora_rank}, alpha={config.lora_alpha}

  DLDL: bins={config.num_age_bins}, σ={config.sigma}

  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {eff_batch}
  LR: {config.learning_rate}
  Epochs: {config.num_epochs}

  === RTX 4090 OPTIMIZATIONS ===
  Gradient Checkpointing: {config.gradient_checkpointing}
  torch.compile: {config.use_compile}
  BF16: {config.use_bf16}
  Fused Optimizer: True
  DataLoader Workers: {config.num_workers}
  Prefetch Factor: {config.prefetch_factor}
  Persistent Workers: {config.persistent_workers}
"""
    )
    print("=" * 70)

    # Train
    print("\n[6/6] Training...")
    print("  (First 1-2 iterations may be slow due to CUDA warmup)\n")

    trainer.train()

    # Final evaluation
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    metrics = trainer.evaluate()
    print(
        f"""
  MAE:  {metrics['eval_mae_months']:.2f} months ({metrics['eval_mae_years']:.2f} years)
  RMSE: {metrics['eval_rmse_months']:.2f} months
  P90:  {metrics['eval_p90_months']:.2f} months
"""
    )

    # Save model
    print("  Saving model...")

    # Handle compiled model (if ever enabled)
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Save adapter
    adapter_path = os.path.join(config.output_dir, "adapter")
    model_to_save.backbone.save_pretrained(adapter_path)

    # Save custom heads
    torch.save(
        {
            "classifier": model_to_save.classifier.state_dict(),
            "gender_embed": model_to_save.gender_embed.state_dict(),
            "pooler": model_to_save.pooler.state_dict(),
        },
        os.path.join(config.output_dir, "heads.pth"),
    )

    # Save config
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "model_id": config.model_id,
                "num_bins": config.num_age_bins,
                "sigma": config.sigma,
                "hidden_size": model_to_save.hidden_size,
                "pooling_type": config.pooling_type,
                "image_size": config.image_size,
                "image_mean": config.image_mean,
                "image_std": config.image_std,
            },
            f,
            indent=2,
        )

    # Save final metrics
    with open(os.path.join(config.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✅ Model saved to: {config.output_dir}")
    print("=" * 70 + "\n")

    return trainer, model


# ==========================================
# 14. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
