import cv2
import numpy as np

class AnomalyDetector:
    """Anomali Tespiti ve Isı Haritası (Heatmap) Üretici Sınıf."""

    def __init__(self, anomaly_threshold: float = 0.15):
        self.anomaly_threshold = anomaly_threshold

    def compute_heatmap(self, image: np.ndarray) -> dict:
        """Görüntü üzerindeki anomali bölgesini ve ısı haritasını hesaplar."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Kenar ve doku sapmalarını yakalama (Laplacian + Blur farkı)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blurred)
        
        # Kenar şiddetini ekle
        edges = cv2.Canny(gray, 50, 150)
        combined_diff = cv2.addWeighted(diff, 0.7, edges, 0.3, 0)

        norm_diff = cv2.normalize(combined_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Heatmap oluşturma
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        
        # Anomali skoru hesaplama (Piksel parlaklık yoğunluğu)
        anomaly_score = float(np.percentile(norm_diff, 95) / 255.0)
        is_anomaly = anomaly_score > self.anomaly_threshold

        base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        overlay = cv2.addWeighted(base_img, 0.5, heatmap, 0.5, 0)

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomaly": is_anomaly,
            "heatmap": heatmap,
            "overlay": overlay
        }