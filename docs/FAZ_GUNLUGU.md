# Faz Günlüğü — AhşapGöz

## Faz 0 — Hazırlık, Ortam ve GitHub Bağlantısı
**Tarih:** 21.10.2026
**Amaç:** Projenin dizin yapısını, sanal ortamını (`venv`), konfigürasyon dosyalarını ve GitHub versiyon kontrol entegrasyonunu kurmak.
**Yapılan işler (adım adım):**
1. Python 3.11 ve NVIDIA RTX 3050 GPU donanım ortamı doğrulandı.
2. Proje dizin yapısı (`src/`, `configs/`, `data/`, `docs/`, `app/` vb.) tam modüler yapıda oluşturuldu.
3. `venv` sanal ortamı kurularak bağımlılıklar (`requirements.txt` ve PyTorch CUDA) yüklendi.
4. `.gitignore`, `README.md`, `LICENSE` ve `config.yaml` oluşturuldu.
5. GitHub reposu bağlandı ve `chore: proje iskeleti ve ortam kurulumu` başlığıyla ilk push gerçekleştirildi.
**Kullanılan yöntem/teknoloji ve gerekçesi:**
- **PyTorch (CUDA 12.1):** RTX 3050 ekran kartı üzerinde derin öğrenme modellerini eğitmek ve çıkarım sürelerini hızlandırmak için seçildi.
- **FastAPI / OpenCV / Ultralytics:** Projenin web, görüntü işleme ve YOLO katmanları için temel kütüphaneler olarak eklendi.
**Çıktılar:**
- Proje klasör ağacı
- GitHub Reposu (v1.0.0 başlangıcı)
**Git Commit:** `chore: proje iskeleti, konfigurasyonlar ve ortam kurulumu`


## Faz 1 — Veri Seti Edinme ve Keşifsel Veri Analizi (EDA)
**Tarih:** 21.10.2026
**Amaç:** Ahşap yüzey kusur veri setini incelemek, sınıf dağılımlarını analiz etmek ve raporda kullanılacak EDA grafiklerini üretmek.
**Yapılan işler:**
1. `scripts/download_data.py` betiği ile veri dizin mimarisi ve YOLO `data.yaml` yapısı hazırlandı.
2. `scripts/run_eda.py` ile veri setindeki 5 temel kusur sınıfının (`canli_budak`, `olu_budak`, `catlak`, `recine`, `mavi_leke`) dağılım analizi yapıldı.
3. Sınıf dağılımı ve kutu (BBox) boyut/en-boy oranı grafikleri üretilerek `docs/figures/` dizinine kaydedildi.
4. `docs/VERI_SETI_KAYNAKLARI.md` tablosu güncellendi.
**Çıktılar:**
- `docs/figures/01_sinif_dagilimi.png`
- `docs/figures/02_bbox_boyut_dagilimi.png`
- `docs/VERI_SETI_KAYNAKLARI.md`
**Git Commit:** `feat(eda): veri seti indirme yapisi ve EDA analiz grafikleri eklendi`


## Faz 2 — Veri Birleştirme, Temizleme ve Hazırlama
**Tarih:** 21.10.2026
**Amaç:** Farklı etiket isimlerini standart Türkçe sınıflara dönüştürmek, YOLO ve Anomali tespiti modelleri için ayrı ayrı işlenmiş veri klasör yapılarını oluşturmak.
**Yapılan işler:**
1. `configs/class_map.yaml` ile sınıf isimleri standartlaştırıldı (`canli_budak`, `olu_budak`, `catlak`, `recine`, `mavi_leke`).
2. `scripts/prepare_dataset.py` geliştirilerek `data/processed/yolo/` altında `train`, `val`, `test` klasör mimarisi otomatikleştirildi.
3. Faz 5'te kullanılacak PatchCore Anomali Tespiti modeli için `data/processed/anomaly/` yapısı (`train_good`, `test_good`, `test_defect`) hazırlandı.
4. YOLO eğitimi için `data/processed/yolo/data.yaml` yapılandırma dosyası üretildi.
**Çıktılar:**
- `configs/class_map.yaml`
- `scripts/prepare_dataset.py`
- `data/processed/yolo/data.yaml`
**Git Commit:** `feat(data): veri temizleme, sinif haritalama ve bölme boru hatti eklendi`


## Faz 3 — Klasik Görüntü İşleme Hattı (Baseline)
**Tarih:** 21.10.2026
**Amaç:** Dersin çekirdek konuları olan klasik görüntü işleme teknikleriyle temel kusur tespit hattını (Baseline) kurmak.
**Yapılan işler:**
1. `src/ahsapgoz/preprocessing.py` modülü altında Gri tonlama, CLAHE ve Gauss filtreleme işlevleri yazıldı.
2. `src/ahsapgoz/classical.py` modülünde Otsu eşikleme, morfolojik açma/kapama ve kontur analizi ile kural tabanlı kusur aday çıkarma algoritması geliştirildi.
3. `scripts/run_classical.py` çalıştırılarak ilk kural tabanlı tespit çıktısı `docs/figures/03_klasik_isleme_sonuc.png` olarak kaydedildi.
**Çıktılar:**
- `src/ahsapgoz/preprocessing.py`
- `src/ahsapgoz/classical.py`
- `docs/figures/03_klasik_isleme_sonuc.png`
**Git Commit:** `feat(classical): CLAHE, morfolojik filtreler ve kontur tabanli klasik tespit motoru eklendi`


## Faz 4 — Derin Öğrenme ile Kusur Tespiti (YOLO)
**Tarih:** 21.10.2026
**Amaç:** YOLO nesne tespiti mimarisini entegre ederek ahşap kusurlarını yüksek doğrulukla konumlandıran ve sınıflandıran derin öğrenme katmanını kurmak.
**Yapılan işler:**
1. `src/ahsapgoz/detector.py` modülü altında Ultralytics YOLOv8 sarmalayıcı sınıfı yazıldı.
2. CUDA / GPU ivmelendirmesi otomatik algılanacak şekilde yapılandırıldı.
3. `scripts/run_yolo.py` ile test çıkarımı gerçekleştirilip sonuç görseli `docs/figures/04_yolo_tespit_sonuc.png` adresine kaydedildi.
**Çıktılar:**
- `src/ahsapgoz/detector.py`
- `scripts/run_yolo.py`
- `docs/figures/04_yolo_tespit_sonuc.png`
**Git Commit:** `feat(yolo): YOLOv8 nesne tespiti sarmalayicisi ve cikarim boru hatti eklendi`


## Faz 5 — Anomali Tespiti Katmanı
**Tarih:** 21.10.2026
**Amaç:** Etiketsiz/bilinmeyen ahşap yüzey kusurlarını yakalamak amacıyla anomali tespiti ve ısı haritası (Heatmap) üretici katmanı entegre etmek.
**Yapılan işler:**
1. `src/ahsapgoz/anomaly.py` modülü geliştirilerek anomali skoru hesaplama ve `COLORMAP_JET` tabanlı ısı haritası renklendirmesi yazıldı.
2. `scripts/run_anomaly.py` çalıştırılarak anomali ısı haritası `docs/figures/05_anomali_isi_haritasi.png` adresine kaydedildi.
**Çıktılar:**
- `src/ahsapgoz/anomaly.py`
- `scripts/run_anomaly.py`
- `docs/figures/05_anomali_isi_haritasi.png`
**Git Commit:** `feat(anomaly): anomali tespiti ve isi haritasi (heatmap) motoru eklendi`


## Faz 6 — Analiz Motoru ve Bilgi Tabanı
**Tarih:** 22.10.2026
**Amaç:** Tespit edilen kusurları endüstriyel metriklerle analiz etmek, Türkçe bilgi tabanından açıklamalar getirmek ve kalite sınıfı (A/B/C/Red) atamak.
**Yapılan işler:**
1. `knowledge/kusur_bilgi_tabani.json` ile kusur türlerine ait Türkçe açıklamalar, oluşum nedenleri, yapısal/estetik etkileri ve çözüm önerileri tanımlandı.
2. `src/ahsapgoz/analysis.py` modülünde kusurlu alan yüzdesi ve çatlak varlığına göre A, B, C veya Red kararı veren kalite motoru yazıldı.
3. `tests/test_analysis.py` ile `pytest` otomasyon testleri yazılarak sistem doğrulandı.
**Çıktılar:**
- `knowledge/kusur_bilgi_tabani.json`
- `src/ahsapgoz/knowledge_base.py`
- `src/ahsapgoz/analysis.py`
- `tests/test_analysis.py`
**Git Commit:** `feat(analysis): kalite siniflandirma motoru, bilgi tabani ve birim testleri eklendi`


## Faz 7 — Web Uygulaması (Backend + Yüksek Kaliteli Arayüz)
**Tarih:** 22.10.2026
**Amaç:** Kullanıcıların ahşap fotoğraflarını yükleyerek canlı analiz alabileceği FastAPI tabanlı web uygulamasını ve ahşap temalı arayüzü kurmak.
**Yapılan işler:**
1. `app/main.py` modülü altında FastAPI rotaları (`GET /`, `POST /analyze`) ve statik dosya servisleri tanımlandı.
2. `app/templates/` altında responsive, ahşap temalı `index.html` ve `result.html` Jinja2 şablonları yazıldı.
3. Görsel yükleme sonrası YOLO kutuları, Anomali Isı Haritası ve Türkçe kusur detay tablosu yan yana sunuldu.
4. `tests/test_api.py` birim testi yazıldı ve doğrulandı.
**Çıktılar:**
- `app/main.py`
- `app/templates/`
- `app/static/`
- `tests/test_api.py`
**Git Commit:** `feat(app): FastAPI backend, ahsap temali HTML/CSS arayuz ve web servisi eklendi`