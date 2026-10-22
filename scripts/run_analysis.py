import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.analysis import DefectAnalyzer

def run_demo():
    print("=== AhşapGöz Analiz Motoru ve Kalite Hesabı Test Ediliyor ===")

    # Örnek Tespitler
    mock_detections = [
        {"label": "canli_budak", "confidence": 0.88, "bbox": [50, 50, 40, 40]},
        {"label": "olu_budak", "confidence": 0.75, "bbox": [150, 200, 30, 30]}
    ]
    surface_area = 400 * 400

    analyzer = DefectAnalyzer()
    result = analyzer.calculate_quality_grade(mock_detections, surface_area)

    print(f" Kalite Sınıfı: {result['grade']}")
    print(f" Açıklama: {result['grade_explanation']}")
    print(f" Toplam Kusur Sayısı: {result['total_defects']}")
    print(f" Kusurlu Alan Oranı: %{result['total_defect_ratio_pct']}")

if __name__ == "__main__":
    run_demo()