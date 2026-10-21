import sys
from pathlib import Path

# Python arama yoluna src dizinini ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import cv2
import numpy as np
from ahsapgoz.detector import YOLODetector

FIGURES_DIR = BASE_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_yolo_demo():
    print("=== YOLO Derin Öğrenme Tespiti Test Ediliyor ===")
    
    detector = YOLODetector(conf_threshold=0.20)

    # Sentetik Test Ahşap Görüntüsü
    synthetic_image = np.full((400, 400, 3), 180, dtype=np.uint8)
    cv2.circle(synthetic_image, (150, 150), 35, (30, 30, 30), -1)
    cv2.line(synthetic_image, (250, 100), (270, 300), (10, 10, 10), 4)

    results = detector.predict(synthetic_image)

    out_path = FIGURES_DIR / "04_yolo_tespit_sonuc.png"
    cv2.imwrite(str(out_path), results["annotated_image"])

    print(f" YOLO Çıkarımı Tamamlandı (Cihaz: {detector.device.upper()})!")
    print(f" Tespit Edilen Nesne Sayısı: {len(results['detections'])}")
    print(f" Sonuç Görseli Kaydedildi: {out_path}")

if __name__ == "__main__":
    run_yolo_demo()