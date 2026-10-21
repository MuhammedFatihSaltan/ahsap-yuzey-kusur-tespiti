# AhşapGöz: Ahşap ve Kereste Üretim Hatlarında Yapay Zekâ Destekli Yüzey Kusuru ve Anomali Tespit Sistemi

Ahşap ve kereste yüzeylerindeki kusurları (budak, çatlak, reçine, leke vb.) klasik görüntü işleme, YOLO ve anomali tespiti ile analiz eden; kalite sınıfı (A/B/C/Red) hesaplayıp PDF rapor üreten Türkçe web uygulaması.

## Özellikler
- **Klasik Görüntü İşleme:** CLAHE, Filtreleme, Otsu/Adaptif Eşikleme, Morfolojik Analiz, Kontur Tespiti.
- **Derin Öğrenme:** YOLOv8/YOLO11 Nesne Tespiti.
- **Anomali Tespiti:** Etiketsiz/bilinmeyen kusurlar için anomali skoru ve ısı haritası (Heatmap).
- **Raporlama:** Otomatik Türkçe PDF kalite kontrol raporu indirimi.

## Kurulum
1. Sanal ortam oluşturun ve aktifleştirin:
   ```cmd
   python -m venv venv
   venv\Scripts\activate