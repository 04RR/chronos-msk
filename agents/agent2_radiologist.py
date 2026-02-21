import joblib
import torch
import numpy as np
import cv2
import os
from PIL import Image
from transformers import SiglipVisionModel, AutoProcessor

class RadiologistAgent:
    def __init__(self, svm_path, model_id="google/medsiglip-448", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"🩺 Radiologist: Loading SVM from {svm_path}...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Suppress text-tower warnings by loading vision model directly
        self.vision_model = (
            SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        )
        self.svm = joblib.load(svm_path)

    def _load_image(self, input_data):
        """
        Robustly loads an image from a path or passes through a numpy array.
        Handles legacy medical formats (TIFF/DICOM-in-JPG) via PIL fallback.
        """
        # Case 1: Input is a File Path
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Image not found at {input_data}")
            
            # Try OpenCV first (Fastest)
            img_bgr = cv2.imread(input_data)
            
            # Fallback: If OpenCV fails (returns None), use PIL
            if img_bgr is None:
                try:
                    pil_img = Image.open(input_data).convert('RGB')
                    img_bgr = np.array(pil_img)[:, :, ::-1] # RGB -> BGR
                except Exception as e:
                    raise ValueError(f"Could not decode image {input_data}: {e}")
            
            return img_bgr

        # Case 2: Input is already a Numpy Array (BGR)
        elif isinstance(input_data, np.ndarray):
            return input_data
        
        else:
            raise TypeError(f"Unsupported input type: {type(input_data)}")

    def _get_embedding(self, img_bgr):
        """Extracts features from a single BGR image."""
        # Ensure standard size/format for SigLIP
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.vision_model(**inputs).pooler_output.cpu().numpy()[0]
        return emb

    def predict(self, input_data):
        """
        Universal Prediction Endpoint.
        Accepts: File Path (str) OR Image Array (np.ndarray)
        Returns: Predicted Stage (str), Embedding (np.array)
        """
        # 1. Standardize Input (Load if path, Pass if array)
        img_bgr = self._load_image(input_data)

        # 2. TTA: Embed Original & Flipped
        emb_orig = self._get_embedding(img_bgr)
        emb_flip = self._get_embedding(cv2.flip(img_bgr, 1))

        # 3. SVM Probabilities
        probs_orig = self.svm.predict_proba([emb_orig])[0]
        probs_flip = self.svm.predict_proba([emb_flip])[0]

        # 4. Average & Argmax
        avg_probs = (probs_orig + probs_flip) / 2.0
        best_idx = np.argmax(avg_probs)
        stage = self.svm.classes_[best_idx]

        # Return Original Embedding for retrieval (to avoid 'mirror world' artifacts)
        return stage, emb_orig