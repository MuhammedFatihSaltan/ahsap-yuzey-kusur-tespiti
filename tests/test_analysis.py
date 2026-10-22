import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.analysis import DefectAnalyzer

def test_quality_grade_a():
    analyzer = DefectAnalyzer()
    detections = [{"label": "canli_budak", "confidence": 0.9, "bbox": [10, 10, 10, 10]}]
    res = analyzer.calculate_quality_grade(detections, 100000)
    assert res["grade"] == "A Sınıfı"

def test_quality_grade_red_on_crack():
    analyzer = DefectAnalyzer()
    detections = [{"label": "catlak", "confidence": 0.9, "bbox": [10, 10, 10, 10]}]
    res = analyzer.calculate_quality_grade(detections, 100000)
    assert res["grade"] == "Red"