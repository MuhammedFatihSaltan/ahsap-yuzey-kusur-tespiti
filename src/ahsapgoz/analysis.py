import numpy as np
from ahsapgoz.knowledge_base import KnowledgeBase

class DefectAnalyzer:
    """Kusur Analiz ve Kalite Sınıflandırma Motoru."""

    def __init__(self, kb_path: str = None):
        self.kb = KnowledgeBase(kb_path)

    def _determine_severity(self, ratio: float, label: str) -> str:
        """Kusur şiddetini belirler."""
        if ratio > 0.05 or label == "anomali_kusur":
            return "Yüksek"
        elif ratio > 0.02:
            return "Orta"
        return "Düşük"

    def _determine_grade(self, defect_ratio: float, labels: list) -> tuple:
        """Kalite sınıfını ve açıklamasını belirler."""
        has_crack = "catlak" in labels or "catlak_adayi" in labels

        if has_crack or defect_ratio > 0.12:
            return "Red", "Çatlak tespiti veya aşırı kusur oranı (%12+) nedeniyle reddedildi."
        elif defect_ratio > 0.05 or "anomali_kusur" in labels:
            return "C Sınıfı", "Yüzeyde yapısal anomali/kusur tespit edildi. Manuel inceleme önerilir."
        elif defect_ratio > 0.02 or "olu_budak" in labels:
            return "B Sınıfı", "Orta düzey kusur oranı (%2 - %5 arası). Genel marangozluk için uygun."
        return "A Sınıfı", "Yüksek kaliteli yüzey (%2 altında kusur). Birinci sınıf mobilya için uygun."

    def calculate_quality_grade(self, detections: list, total_surface_area: int, anomaly_res: dict = None) -> dict:
        """Kusur tespitlerine göre toplam kalite sınıfı (A/B/C/Red) ve metrikleri hesaplar."""
        if total_surface_area <= 0:
            total_surface_area = 1

        is_anomaly = anomaly_res.get("is_anomaly", False) if anomaly_res else False
        anomaly_score = anomaly_res.get("anomaly_score", 0.0) if anomaly_res else 0.0

        if len(detections) == 0 and is_anomaly:
            detections.append({
                "label": "anomali_kusur",
                "confidence": anomaly_score,
                "bbox": [100, 100, 150, 150]
            })

        total_defect_area = sum([d.get("bbox", [0, 0, 0, 0])[2] * d.get("bbox", [0, 0, 0, 0])[3] for d in detections])
        defect_ratio = float(total_defect_area / total_surface_area)

        labels = [d.get("label", "") for d in detections]
        grade, grade_explanation = self._determine_grade(defect_ratio, labels)

        analyzed_defects = []
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]
            ratio = float(area / total_surface_area)
            label = d.get("label", "bilinmeyen")

            severity = self._determine_severity(ratio, label)
            
            if label == "anomali_kusur":
                info = {
                    "ad": "Bilinmeyen Yüzey Anomalisi (Anomali Tespiti)",
                    "onerilen_islem": "Yüzeydeki renk/doku düzensizliği nedeniyle uzman kontrolü gereklidir."
                }
            else:
                info = self.kb.get_info(label)

            analyzed_defects.append({
                "label": label,
                "confidence": d.get("confidence", 0.0),
                "bbox": bbox,
                "area_pixel": area,
                "surface_ratio_pct": round(ratio * 100, 2),
                "severity": severity,
                "details": info
            })

        return {
            "grade": grade,
            "grade_explanation": grade_explanation,
            "total_defects": len(detections),
            "total_defect_ratio_pct": round(defect_ratio * 100, 2),
            "defects": analyzed_defects
        }