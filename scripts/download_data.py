import os
import sys
import logging
from pathlib import Path
import yaml

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def check_or_create_dirs():
    """Gerekli veri klasörlerini oluşturur."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Veri dizini hazır: {RAW_DATA_DIR}")

def create_sample_dataset_structure():
    """
    Eğer Roboflow API anahtarı yoksa veya indirme başarısız olursa
    EDA ve veri boru hattının test edilebilmesi için yapı oluşturur.
    """
    logger.info("Veri seti yapısı doğrulanıyor...")
    images_dir = RAW_DATA_DIR / "images"
    labels_dir = RAW_DATA_DIR / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # data.yaml tanım dosyası
    yaml_content = {
        "path": str(RAW_DATA_DIR.absolute()),
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
    
    yaml_path = RAW_DATA_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"YOLO konfigürasyon dosyası oluşturuldu: {yaml_path}")

def download_roboflow_dataset(api_key: str = None):
    """Roboflow API kullanarak veri setini indirir."""
    if not api_key:
        logger.warning("Roboflow API Key bulunamadı. Lütfen .env dosyasında ROBOFLOW_API_KEY tanımlayın.")
        create_sample_dataset_structure()
        return

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        # Örnek Roboflow ahşap kusur projesi
        project = rf.workspace("tim-4ijf0").project("wood-surface-defect-detection")
        dataset = project.version(1).download("yolov8", location=str(RAW_DATA_DIR))
        logger.info("Roboflow veri seti başarıyla indirildi!")
    except Exception as e:
        logger.error(f"Roboflow üzerinden veri indirilirken hata oluştu: {e}")
        logger.info("Alternatif yapı oluşturuluyor...")
        create_sample_dataset_structure()

if __name__ == "__main__":
    check_or_create_dirs()
    api_key = os.getenv("ROBOFLOW_API_KEY", None)
    download_roboflow_dataset(api_key)