import sys
from pathlib import Path

# Python arama yoluna src dizinini ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import cv2
import numpy as np
from ahsapgoz.anomaly import AnomalyDetector

FIGURES_DIR = BASE_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_demo():
    print("=== Anomali Tespiti Katmanı Test Ediliyor ===")

    # Test Ahşap Görüntüsü
    synthetic_image = np.full((400, 400, 3), 180, dtype=np.uint8)
    cv2.circle(synthetic_image, (200, 200), 40, (30, 30, 30), -1)

    detector = AnomalyDetector()
    results = detector.compute_heatmap(synthetic_image)

    out_path = FIGURES_DIR / "05_anomali_isi_haritasi.png"
    cv2.imwrite(str(out_path), results["overlay"])

    print(" Anomali Analizi Tamamlandı!")
    print(f" Anomali Skoru: {results['anomaly_score']}")
    print(f" Anomali Var mı?: {results['is_anomaly']}")
    print(f" Isı Haritası Kaydedildi: {out_path}")

if __name__ == "__main__":
    run_demo()