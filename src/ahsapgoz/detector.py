import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLODetector:
    """YOLO Nesne Tespiti Sarmalayıcı Sınıfı."""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Ön eğitilmiş nano model yüklenir
            self.model = YOLO("yolov8n.pt")
            
        self.class_names = {
            0: "canli_budak",
            1: "olu_budak",
            2: "catlak",
            3: "recine",
            4: "mavi_leke"
        }

    def predict(self, image: np.ndarray) -> dict:
        """Görüntü üzerinde kusur tespiti yapar."""
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        annotated_image = image.copy()

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            label_name = self.class_names.get(cls_id, f"kusur_{cls_id}")

            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "class_id": cls_id,
                "label": label_name,
                "confidence": round(conf, 3)
            })

            # Görsel üzerine kutu ve etiket çizimi
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label_name} {conf:.2f}"
            cv2.putText(annotated_image, caption, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return {
            "detections": detections,
            "annotated_image": annotated_image
        }