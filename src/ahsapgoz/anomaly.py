import cv2
import numpy as np

class AnomalyDetector:
    """Anomali Tespiti ve Isı Haritası (Heatmap) Üretici Sınıf."""

    def __init__(self, anomaly_threshold: float = 0.35):
        self.anomaly_threshold = anomaly_threshold

    def compute_heatmap(self, image: np.ndarray) -> tuple:
        """Görüntü üzerindeki anomali bölgesini ve ısı haritasını hesaplar."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Temiz doku referans sapması simülasyonu
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blurred)
        norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        
        # Heatmap oluşturma
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        
        # Anomali skoru hesaplama
        anomaly_score = float(np.mean(norm_diff) / 255.0)
        is_anomaly = anomaly_score > self.anomaly_threshold

        # Orijinal görsel ile heatmap harmanlama
        if len(image.shape) == 2:
            base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            base_img = image.copy()
            
        overlay = cv2.addWeighted(base_img, 0.6, heatmap, 0.4, 0)

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomaly": is_anomaly,
            "heatmap": heatmap,
            "overlay": overlay
        }