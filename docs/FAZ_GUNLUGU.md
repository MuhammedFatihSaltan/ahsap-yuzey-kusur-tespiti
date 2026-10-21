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