import numpy as np
from ahsapgoz.knowledge_base import KnowledgeBase


class DefectAnalyzer:
    """Kusur Analiz ve Kalite Sınıflandırma Motoru."""

    def __init__(self, kb_path: str = None, iou_threshold: float = 0.45):
        self.kb = KnowledgeBase(kb_path)
        self.iou_threshold = iou_threshold

    def _calculate_iou(self, box1: list, box2: list) -> float:
        """İki [x, y, w, h] kutusu arasındaki kesişim/birleşim (IoU) oranını hesaplar."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        inter_area = max(0, xb - xa) * max(0, yb - ya)
        if inter_area == 0:
            return 0.0

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = float(box1_area + box2_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    def _filter_overlapping_detections(self, detections: list) -> list:
        """Aynı bölgeyi gösteren üst üste binmiş kutulardan sadece en yüksek güven skorlusunu tutar."""
        if not detections:
            return []

        # Güven skoruna göre büyükten küçüğe sırala
        sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0.0), reverse=True)
        filtered = []

        for current in sorted_dets:
            curr_box = current.get("bbox", [0, 0, 0, 0])
            is_duplicate = False
            for kept in filtered:
                kept_box = kept.get("bbox", [0, 0, 0, 0])
                if self._calculate_iou(curr_box, kept_box) > self.iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(current)

        return filtered

    def _calculate_union_area(self, detections: list, total_surface_area: int) -> int:
        """Kısmen örtüşen kutuların ortak alanlarını iki kez saymadan gerçek birleşim alanını hesaplar."""
        if not detections:
            return 0

        # Kutu koordinatlarından maksimum sınırları bul
        max_x = max([int(d["bbox"][0] + d["bbox"][2]) for d in detections] + [1])
        max_y = max([int(d["bbox"][1] + d["bbox"][3]) for d in detections] + [1])

        mask = np.zeros((max_y, max_x), dtype=np.uint8)
        for d in detections:
            x, y, w, h = [int(v) for v in d.get("bbox", [0, 0, 0, 0])]
            mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = 1

        union_pixels = int(np.sum(mask))
        return min(union_pixels, total_surface_area)

    def _determine_severity(self, ratio: float, label: str) -> str:
        """Kusur şiddetini belirler."""
        if ratio > 0.05:
            return "Yüksek"
        elif ratio > 0.01 or label == "anomali_kusur":
            return "Orta"
        return "Düşük"

    def _determine_grade(self, defect_ratio: float, labels: list) -> tuple:
        """Kalite sınıfını ve açıklamasını belirler."""
        has_crack = "catlak" in labels or "catlak_adayi" in labels

        if has_crack or defect_ratio > 0.12:
            return "Red", "Çatlak tespiti veya aşırı kusur oranı (%12+) nedeniyle reddedildi."
        elif defect_ratio > 0.05:
            return "C Sınıfı", "Yüzeyde belirgin yapısal anomali/kusur tespit edildi (%5 - %12). Manuel inceleme önerilir."
        elif defect_ratio > 0.02 or "olu_budak" in labels or "anomali_kusur" in labels:
            return "B Sınıfı", "Orta düzey kusur veya yüzey deformasyonu tespit edildi. Genel marangozluk/kaplama öncesi işlem için uygun."
        return "A Sınıfı", "Yüksek kaliteli yüzey (%2 altında kusur). Birinci sınıf mobilya için uygun."

    def calculate_quality_grade(self, detections: list, total_surface_area: int, anomaly_res: dict = None) -> dict:
        """Kusur tespitlerine göre toplam kalite sınıfı (A/B/C/Red) ve metrikleri hesaplar."""
        if total_surface_area <= 0:
            total_surface_area = 1

        raw_detections = list(detections) if detections else []
        is_anomaly = anomaly_res.get("is_anomaly", False) if anomaly_res else False
        anomaly_score = anomaly_res.get("anomaly_score", 0.0) if anomaly_res else 0.0
        anomaly_regions = anomaly_res.get("regions", []) if anomaly_res else []

        # 1. Üst üste binen aynı kusur tahminlerini teke indir (NMS)
        clean_detections = self._filter_overlapping_detections(raw_detections)

        # 2. YOLO tespit bulamadıysa anomali dedektörünün bulduğu gerçek bölgeleri kullan
        if len(clean_detections) == 0 and is_anomaly:
            if anomaly_regions:
                clean_detections.extend(self._filter_overlapping_detections(anomaly_regions))
            else:
                side = int(np.sqrt(total_surface_area) * 0.1)
                clean_detections.append({
                    "label": "anomali_kusur",
                    "confidence": anomaly_score,
                    "bbox": [0, 0, side, side]
                })

        # 3. Örtüşen alanları çift saymadan gerçek birleşim alanını hesapla
        total_defect_area = self._calculate_union_area(clean_detections, total_surface_area)
        defect_ratio = float(min(total_defect_area / total_surface_area, 1.0))

        labels = [d.get("label", "") for d in clean_detections]
        grade, grade_explanation = self._determine_grade(defect_ratio, labels)

        analyzed_defects = []
        for d in clean_detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]
            ratio = float(min(area / total_surface_area, 1.0))
            label = d.get("label", "bilinmeyen")

            severity = self._determine_severity(ratio, label)

            if label == "anomali_kusur":
                info = {
                    "ad": "Yüzey Anomalisi / Kaplama Deformasyonu",
                    "onerilen_islem": "Yüzeydeki renk/kaplama kalkması nedeniyle zımpara veya dolgu/rötuş işlemi önerilir."
                }
            else:
                info = self.kb.get_info(label)

            analyzed_defects.append({
                "label": label,
                "confidence": d.get("confidence", 0.0),
                "bbox": bbox,
                "area_pixel": int(area),
                "surface_ratio_pct": round(ratio * 100, 2),
                "severity": severity,
                "details": info
            })

        analyzed_defects.sort(key=lambda x: x["area_pixel"], reverse=True)

        return {
            "grade": grade,
            "grade_explanation": grade_explanation,
            "total_defects": len(analyzed_defects),
            "total_defect_ratio_pct": round(defect_ratio * 100, 2),
            "defects": analyzed_defects,
            "filtered_detections": clean_detections
        }