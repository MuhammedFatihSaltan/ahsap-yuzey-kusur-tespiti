import os
import shutil
from pathlib import Path
import yaml
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_FILE = BASE_DIR / "configs" / "class_map.yaml"

def setup_directories():
    """YOLO ve Anomali modeli için hedef dizinleri temiz bir şekilde hazırlar."""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
        
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Anomali Mimarisi Dizinleri
    (PROCESSED_DIR / "anomaly" / "train_good").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "anomaly" / "test_good").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "anomaly" / "test_defect").mkdir(parents=True, exist_ok=True)

    logger.info("Veri işleme dizin yapısı oluşturuldu.")

def generate_processed_data_yaml():
    """Eğitim için kullanılacak nihai data.yaml dosyasını yazar."""
    data_yaml = {
        "path": str((PROCESSED_DIR / "yolo").absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "canli_budak",
            1: "olu_budak",
            2: "catlak",
            3: "recine",
            4: "mavi_leke"
        }
    }
    
    out_path = PROCESSED_DIR / "yolo" / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Nihai YOLO konfigürasyonu kaydedildi: {out_path}")

def prepare_pipeline():
    print("=== AhşapGöz Veri Hazırlama ve Bölme Hattı ===")
    setup_directories()
    generate_processed_data_yaml()
    
    print("\n[BILGI] Veri birleştirme ve bölme yapılandırması tamamlandı.")
    print(f" İşlenmiş Veri Konumu: {PROCESSED_DIR.absolute()}")

if __name__ == "__main__":
    prepare_pipeline()