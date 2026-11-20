import cv2
import numpy as np


class AnomalyDetector:
    """Ahşap damar dokusunu filtreleyip yalnızca gerçek lokal yüzey kusurlarını tespit eder."""

    def __init__(self, anomaly_threshold: float = 0.15, min_defect_area: int = 35):
        self.anomaly_threshold = anomaly_threshold
        self.min_defect_area = min_defect_area

    def compute_heatmap(self, image: np.ndarray) -> dict:
        """Görüntü üzerindeki gerçek kusur bölgelerini ve ısı haritasını hesaplar."""
        if len(image.shape) == 2:
            bgr_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr_img = image.copy()

        # 1. LAB renk uzayına geç (L: Parlaklık, b: Sarı/Bej tonu)
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l_channel, _, b_channel = cv2.split(lab)

        # 2. Ahşap damarlarını bastırmak için geniş medyan bulanıklaştırma
        l_smooth = cv2.GaussianBlur(l_channel, (5, 5), 0)
        bg_l = cv2.medianBlur(l_smooth, 51)

        # Arka plandan belirgin şekilde açık (boya kalkması/soyulma) veya çok koyu (derin delik/budak) farklar
        bright_diff = cv2.subtract(l_smooth, bg_l)  # Boya soyulmaları (açık krem/bej lekeler)
        dark_diff = cv2.subtract(bg_l, l_smooth)    # Derin çukur/koyu lekeler

        # Renk tonu (b kanalı) sapmasını da hesaba kat
        b_smooth = cv2.GaussianBlur(b_channel, (5, 5), 0)
        bg_b = cv2.medianBlur(b_smooth, 51)
        color_diff = cv2.absdiff(b_smooth, bg_b)

        # Birleşik anomali haritası (açık renkli soyulmalara ve lokal farklara odaklan)
        raw_anomaly = cv2.addWeighted(bright_diff, 0.75, color_diff, 0.25, 0)
        raw_anomaly = np.maximum(raw_anomaly, (dark_diff * 0.6).astype(np.uint8))

        # İnce ahşap damar çizgilerini temizlemek için morfolojik açma (Opening)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_anomaly = cv2.morphologyEx(raw_anomaly, cv2.MORPH_OPEN, clean_kernel)

        norm_diff = cv2.normalize(filtered_anomaly, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)

        # 3. İstatistiksel eşikleme (Sadece yüzey ortalamasından 3 standart sapma sapan gerçek kusurlar)
        mean_val = float(np.mean(filtered_anomaly))
        std_val = float(np.std(filtered_anomaly))
        dynamic_thresh = max(mean_val + 3.0 * std_val, 22.0)

        _, mask = cv2.threshold(filtered_anomaly, dynamic_thresh, 255, cv2.THRESH_BINARY)

        # Yakın parçaları birleştir (Closing)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        anomaly_regions = []
        annotated_anomaly = bgr_img.copy()

        # Görüntü boyutuna göre dinamik minimum alan (çok küçük toz/parlama noktalarını eler)
        img_area = bgr_img.shape[0] * bgr_img.shape[1]
        dynamic_min_area = max(self.min_defect_area, int(img_area * 0.00008))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < dynamic_min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Çerçeve kenarlarındaki uzun ince gölge çizgilerini ele (aşırı uzun/ince oran filtresi)
            aspect_ratio = max(w / max(h, 1), h / max(w, 1))
            if aspect_ratio > 12.0 and area < dynamic_min_area * 4:
                continue

            roi = filtered_anomaly[y:y+h, x:x+w]
            peak_val = float(np.max(roi)) if roi.size > 0 else 0.0
            region_conf = min(max(peak_val / 120.0, 0.45), 0.98)

            anomaly_regions.append({
                "label": "anomali_kusur",
                "confidence": round(region_conf, 3),
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area)
            })
            cv2.rectangle(annotated_anomaly, (x, y), (x + w, y + h), (0, 215, 255), 2)

        anomaly_score = float(np.percentile(norm_diff, 98) / 255.0)
        is_anomaly = len(anomaly_regions) > 0 or (anomaly_score > self.anomaly_threshold and std_val > 10.0)

        overlay = cv2.addWeighted(bgr_img, 0.55, heatmap, 0.45, 0)

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomaly": is_anomaly,
            "heatmap": heatmap,
            "overlay": overlay,
            "annotated_anomaly": annotated_anomaly,
            "regions": anomaly_regions
        }