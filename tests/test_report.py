import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.report import PDFReportGenerator

def test_pdf_generation():
    generator = PDFReportGenerator()
    out_pdf = BASE_DIR / "outputs" / "results" / "test_output.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    mock_data = {
        "grade": "A Sınıfı",
        "grade_explanation": "Test",
        "total_defects": 0,
        "total_defect_ratio_pct": 0.0,
        "defects": []
    }

    generator.generate_report(mock_data, "invalid.jpg", "invalid.jpg", str(out_pdf))
    
    assert os.path.exists(out_pdf)
    assert os.path.getsize(out_pdf) > 0