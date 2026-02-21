import cv2
import numpy as np
from ultralytics import YOLO


class ScoutAgent:
    def __init__(self, model_path):
        print(f"🔭 Scout: Loading YOLO from {model_path}...")
        self.model = YOLO(model_path)

    def predict(self, image_path):
        """
        Input: Path to raw X-ray.
        Output: Cropped BGR image of the Distal Radius (corrected orientation).
        """
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image at {image_path}")

        best_conf = -1.0
        best_crop = None

        # Rotations: 0, 90, 180, 270
        rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]

        for rot_code in rotations:
            img_variant = (
                cv2.rotate(original, rot_code)
                if rot_code is not None
                else original.copy()
            )

            # Run Inference
            results = self.model(img_variant, verbose=False)[0]

            for box in results.boxes:
                # Class 0 = Radius (Modify if your class ID is different)
                if int(box.cls[0]) == 0:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf

                        # Extract Coordinates with Padding
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w = img_variant.shape[:2]
                        pad_x = int((x2 - x1) * 0.15)
                        pad_y = int((y2 - y1) * 0.15)

                        y1_pad = max(0, y1 - pad_y)
                        y2_pad = min(h, y2 + pad_y)
                        x1_pad = max(0, x1 - pad_x)
                        x2_pad = min(w, x2 + pad_x)

                        best_crop = img_variant[y1_pad:y2_pad, x1_pad:x2_pad]

        if best_crop is None:
            raise ValueError("Scout failed to detect Radius in any orientation.")

        return best_crop
