import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.report import PDFReportGenerator

def run_demo():
    print("=== AhşapGöz PDF Rapor Motoru Test Ediliyor ===")

    # Sentetik Test Görselleri Kaydetme
    img_dir = BASE_DIR / "outputs" / "results"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = str(img_dir / "demo_img.jpg")
    heat_path = str(img_dir / "demo_heat.jpg")
    pdf_path = str(img_dir / "demo_rapor.pdf")

    dummy_img = np.full((300, 400, 3), 180, dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    cv2.imwrite(heat_path, dummy_img)

    mock_analysis = {
        "grade": "B Sınıfı",
        "grade_explanation": "Orta düzey kusur oranı (%2 - %5 arası). Genel marangozluk için uygun.",
        "total_defects": 1,
        "total_defect_ratio_pct": 1.5,
        "defects": [
            {
                "confidence": 0.88,
                "surface_ratio_pct": 1.5,
                "severity": "Orta",
                "details": {
                    "ad": "Canlı Budak",
                    "onerilen_islem": "Zımparalama ve yüzey koruyucu vernik uygulaması yeterlidir."
                }
            }
        ]
    }

    generator = PDFReportGenerator()
    out = generator.generate_report(mock_analysis, img_path, heat_path, pdf_path)

    print(f" PDF Raporu Oluşturuldu: {out}")

if __name__ == "__main__":
    run_demo()