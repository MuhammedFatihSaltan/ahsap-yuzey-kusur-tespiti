import cv2
import numpy as np

class ClassicalDefectDetector:
    """Klasik Görüntü İşleme yöntemleri ile kusur tespit sınıfı."""

    def __init__(self, min_area: int = 100, max_area: int = 50000):
        self.min_area = min_area
        self.max_area = max_area

    def apply_threshold(self, image: np.ndarray) -> np.ndarray:
        """Otsu ve Adaptif eşikleme uygular."""
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu

    def apply_morphology(self, binary_image: np.ndarray) -> np.ndarray:
        """Morfolojik Açma ve Kapama işlemleri uygular."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing

    def detect_defects(self, preprocessed_dict: dict, original_image: np.ndarray) -> dict:
        """Kontur analizi yaparak kusur adaylarını tespit eder."""
        denoised = preprocessed_dict["denoised"]
        binary = self.apply_threshold(denoised)
        morph = self.apply_morphology(binary)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated_image = original_image.copy()
        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

                # Basit kural tabanlı sınıflandırma mantığı
                if circularity > 0.5:
                    label = "budak_adayi"
                elif aspect_ratio > 2.0 or aspect_ratio < 0.5:
                    label = "catlak_adayi"
                else:
                    label = "kusur_adayi"

                detections.append({
                    "bbox": [x, y, w, h],
                    "area": area,
                    "label": label,
                    "confidence": 0.65  # Klasik yöntem kural tahmini
                })

                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_image, label, (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return {
            "binary": binary,
            "morph": morph,
            "detections": detections,
            "annotated_image": annotated_image
        }