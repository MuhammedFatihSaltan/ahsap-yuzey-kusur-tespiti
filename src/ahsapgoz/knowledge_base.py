import json
from pathlib import Path

class KnowledgeBase:
    """Kusur Bilgi Tabanı Yükleyici ve Sorgulayıcı."""

    def __init__(self, kb_path: str = None):
        if kb_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent.parent
            kb_path = BASE_DIR / "knowledge" / "kusur_bilgi_tabani.json"
            
        self.kb_path = Path(kb_path)
        self.data = self._load_data()

    def _load_data(self) -> dict:
        if not self.kb_path.exists():
            return {}
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_info(self, defect_type: str) -> dict:
        """Kusur türüne ait detaylı bilgiyi döndürür."""
        return self.data.get(defect_type, {
            "ad": defect_type.capitalize(),
            "kisa_aciklama": "Tanımlanmamış yüzey kusuru.",
            "olusum_nedeni": "Bilinmiyor.",
            "yapisal_etki": "Belirsiz.",
            "estetik_etki": "Belirsiz.",
            "onerilen_islem": "Uzman incelemesi tavsiye edilir.",
            "kullanim_uygunlugu": "Kısıtlı."
        })