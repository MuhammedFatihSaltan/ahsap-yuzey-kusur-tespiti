import numpy as np
from ahsapgoz.knowledge_base import KnowledgeBase

class DefectAnalyzer:
    """Kusur Analiz ve Kalite Sınıflandırma Motoru."""

    def __init__(self, kb_path: str = None):
        self.kb = KnowledgeBase(kb_path)

    def calculate_quality_grade(self, detections: list, total_surface_area: int) -> dict:
        """
        Kusur tespitlerine göre toplam kalite sınıfı (A/B/C/Red) ve metrikleri hesaplar.
        """
        if total_surface_area <= 0:
            total_surface_area = 1

        total_defect_area = sum([d.get("bbox", [0, 0, 0, 0])[2] * d.get("bbox", [0, 0, 0, 0])[3] for d in detections])
        defect_ratio = float(total_defect_area / total_surface_area)

        labels = [d.get("label", "") for d in detections]
        has_crack = "catlak" in labels or "catlak_adayi" in labels

        # Kalite Sınıflandırma Kuralları (Ayarlanabilir İç Ölçüt)
        if has_crack or defect_ratio > 0.12:
            grade = "Red"
            grade_explanation = "Çatlak tespiti veya aşırı kusur oranı (%12+) nedeniyle reddedildi."
        elif defect_ratio > 0.05:
            grade = "C Sınıfı"
            grade_explanation = "Yüksek kusur oranı (%5 - %12 arası). Sadece ambalaj/palet için uygun."
        elif defect_ratio > 0.02 or "olu_budak" in labels:
            grade = "B Sınıfı"
            grade_explanation = "Orta düzey kusur oranı (%2 - %5 arası). Genel marangozluk için uygun."
        else:
            grade = "A Sınıfı"
            grade_explanation = "Yüksek kaliteli yüzey (%2 altında kusur). Birinci sınıf mobilya için uygun."

        # Her kusur için bilgi tabanı zenginleştirmesi
        analyzed_defects = []
        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]
            ratio = float(area / total_surface_area)
            label = d.get("label", "bilinmeyen")

            severity = "Yüksek" if ratio > 0.05 else ("Orta" if ratio > 0.02 else "Düşük")
            
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