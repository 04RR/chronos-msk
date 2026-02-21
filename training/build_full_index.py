import os
import pandas as pd
import numpy as np
import faiss
import torch
import cv2
import json
from tqdm import tqdm
from transformers import SiglipVisionModel, AutoProcessor

# --- CONFIGURATION ---
CSV_PATH = "data/hand_atlas_dataset.csv"
OUTPUT_DIR = "indices_full" # New directory for full-image vectors
MODEL_ID = "google/medsiglip-448"
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "cpu"

def letterbox_resize(image, size=448):
    """Resizes image to square while preserving aspect ratio (padding)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    padded[top : top + new_h, left : left + new_w] = resized
    return padded

def build_index_for_group(group_name, group_df, embed_model, processor):
    print(f"\n🏗️  Building Full-Image Index for: {group_name} ({len(group_df)} images)")
    embeddings = []
    valid_metadata = []

    for _, row in tqdm(group_df.iterrows(), total=len(group_df)):
        path = row["image_path"]
        if not os.path.exists(path): continue

        # 1. LOAD FULL IMAGE (No YOLO/Scout Crop)
        img_bgr = cv2.imread(path)
        if img_bgr is None: continue

        # 2. PREPROCESS
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = letterbox_resize(img_rgb)

        # 3. EMBED
        inputs = processor(images=img_resized, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = embed_model(**inputs).pooler_output
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True) # Normalize

        embeddings.append(feat.cpu().numpy())

        # Save metadata ensuring Age is standardized
        raw_age = float(row["age"])
        # Heuristic: If age < 25, convert to months immediately for consistency
        age_months = raw_age * 12.0 if raw_age < 25.0 else raw_age

        valid_metadata.append({
            "id": str(row["id"]),
            "age_years": raw_age, # Keep original for reference
            "age_months": age_months, # The one we train on
            "gender": row["gender"],
            "race": row["race"],
            "path": path,
        })

    if not embeddings: return

    # 4. SAVE FAISS
    final_embeds = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(final_embeds.shape[1])
    index.add(final_embeds)

    safe_name = group_name.replace(" ", "_")
    faiss.write_index(index, os.path.join(OUTPUT_DIR, f"{safe_name}.index"))
    with open(os.path.join(OUTPUT_DIR, f"{safe_name}_meta.json"), "w") as f:
        json.dump(valid_metadata, f)

    print(f"✅ Saved {safe_name} ({index.ntotal} vectors)")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading MedSigLIP...")
    embed_model = SiglipVisionModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    df = pd.read_csv(CSV_PATH)
    # Cleanup columns
    df.columns = df.columns.str.strip()
    df['gender'] = df['gender'].str.strip()
    df['race'] = df['race'].str.strip()

    groups = df.groupby(["gender", "race"])

    for (gender, race), group_df in groups:
        if race == "White": race = "Caucasian"
        group_key = f"{gender}_{race}"
        build_index_for_group(group_key, group_df, embed_model, processor)

if __name__ == "__main__":
    main()
