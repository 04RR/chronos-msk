import os
import pandas as pd
import numpy as np
import torch
import joblib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from transformers import SiglipVisionModel, AutoProcessor
from tqdm import tqdm

plt.switch_backend("agg")

# --- CONFIGURATION ---
CSV_PATH = "teacher_labels.csv"
IMAGE_DIR = "agent2_inputs/radius_ulna_joint"
MODEL_ID = "google/medsiglip-448"
OUTPUT_MODEL = "radiologist_head.pkl"
CONFUSION_MATRIX_FILE = "agent2_performance.png"


def get_embeddings(df, model, processor, device):
    embeddings = []
    labels = []

    print("Extracting features from Teacher Dataset...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(IMAGE_DIR, row["filename"])
        if not os.path.exists(img_path):
            continue

        # Load & Preprocess
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image, return_tensors="pt").to(device)

        # Inference (Extract features using Agent 3's brain)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.pooler_output.cpu().numpy()[0]

        embeddings.append(emb)
        labels.append(row["stage"])

    return np.array(embeddings), np.array(labels)


def main():
    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        print("❌ Error: teacher_labels.csv not found.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} labels from Auto-Labeler.")

    # 2. Load Vision Backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading MedSigLIP on {device}...")
    model = SiglipVisionModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 3. Generate Embeddings
    X, y = get_embeddings(df, model, processor, device)

    # 4. Train/Test Split (The "Honest" Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # 5. Train SVM
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)}...")
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", probability=True, class_weight="balanced"),
    )
    clf.fit(X_train, y_train)

    # 6. Evaluate
    print("\n--- AGENT 2 PERFORMANCE REPORT ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 7. Generate Confusion Matrix Plot (For your submission)
    cm = confusion_matrix(
        y_test, y_pred, labels=["B", "C", "D", "E", "F", "G", "H", "I"]
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["B", "C", "D", "E", "F", "G", "H", "I"],
        yticklabels=["B", "C", "D", "E", "F", "G", "H", "I"],
    )
    plt.xlabel("Predicted Stage")
    plt.ylabel("Actual Stage (Gemini)")
    plt.title("Agent 2: Radiologist Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_FILE)
    print(f"✅ Confusion Matrix saved to {CONFUSION_MATRIX_FILE}")

    # 8. Save Model
    joblib.dump(clf, OUTPUT_MODEL)
    print(f"✅ Trained Agent saved to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
