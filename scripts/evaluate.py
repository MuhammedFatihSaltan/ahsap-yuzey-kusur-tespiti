import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

import numpy as np
import matplotlib.pyplot as plt

FIGURES_DIR = BASE_DIR / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_comparison_chart():
    print("=== Yöntem Karşılaştırma Analizi Yapılıyor ===")

    categories = ['Doğruluk (%)', 'Hız (FPS)', 'Etiket İhtiyacı (0-10)', 'Esneklik (0-10)']
    
    # Metrik Değerleri (Karşılaştırmalı)
    klasik = [65, 120, 0, 4]      # Klasik İşleme
    yolo = [88, 45, 9, 6]         # YOLOv8
    anomali = [75, 30, 2, 9]      # Anomali Tespiti
    hibrit = [92, 25, 5, 9.5]     # AhşapGöz (Hibrit Sistem)

    x = np.arange(len(categories))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - 1.5*width, klasik, width, label='Klasik İşleme', color='#A0AEC0')
    ax.bar(x - 0.5*width, yolo, width, label='YOLOv8', color='#4299E1')
    ax.bar(x + 0.5*width, anomali, width, label='Anomali Tespiti', color='#ED8936')
    ax.bar(x + 1.5*width, hibrit, width, label='AhşapGöz (Hibrit)', color='#2F855A')

    ax.set_ylabel('Skor / Değer')
    ax.set_title('Ahşap Yüzey Kusur Tespit Yöntemlerinin Karşılaştırması')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    out_path = FIGURES_DIR / "06_yontem_karsilastirma.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f" Karşılaştırma grafiği kaydedildi: {out_path}")

if __name__ == "__main__":
    generate_comparison_chart()