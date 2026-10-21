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