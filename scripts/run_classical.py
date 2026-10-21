import sys
from pathlib import Path

# Python arama yoluna src dizinini ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import cv2
import numpy as np
from ahsapgoz.preprocessing import ImagePreprocessor
from ahsapgoz.classical import ClassicalDefectDetector

FIGURES_DIR = BASE_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_demo():
    print("=== Klasik Görüntü İşleme Hattı Test Ediliyor ===")
    
    # Sentetik Test Ahşap Görüntüsü Oluşturma
    synthetic_image = np.full((400, 400, 3), 180, dtype=np.uint8) # Ahşap zemin
    cv2.circle(synthetic_image, (150, 150), 30, (40, 40, 40), -1) # Budak simülasyonu
    cv2.line(synthetic_image, (250, 100), (270, 300), (20, 20, 20), 3) # Çatlak simülasyonu

    preprocessor = ImagePreprocessor()
    detector = ClassicalDefectDetector()

    prep_results = preprocessor.preprocess(synthetic_image)
    detect_results = detector.detect_defects(prep_results, synthetic_image)

    out_path = FIGURES_DIR / "03_klasik_isleme_sonuc.png"
    cv2.imwrite(str(out_path), detect_results["annotated_image"])
    
    print(f" Klasik tespit tamamlandı!")
    print(f" Bulunan Kusur Sayısı: {len(detect_results['detections'])}")
    print(f" Sonuç Görseli Kaydedildi: {out_path}")

if __name__ == "__main__":
    run_demo()