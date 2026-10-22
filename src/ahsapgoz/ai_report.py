import os
import time
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class AIReportGenerator:
    """Google Gemini API kullanarak ahşap yüzey analiz verilerini detaylı Türkçe rapora dönüştürür."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
        
        self.config = self._load_config(config_path)
        self.api_key = os.getenv("GEMINI_API_KEY")
        
    def _load_config(self, config_path: Path) -> dict:
        try:
            if Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    return cfg.get("ai_service", {})
        except Exception as e:
            logger.warning("Config yüklenirken hata oluştu: %s", e)
        return {
            "provider": "google",
            "model": "gemini-3.8-flash",
            "max_tokens": 1000,
            "timeout_seconds": 30
        }

    def _build_prompt(self, analysis_result: dict) -> str:
        defects_str = ""
        for idx, d in enumerate(analysis_result.get("defects", []), 1):
            details = d.get("details", {})
            defects_str += f"{idx}. Tür: {details.get('ad', d.get('label'))}, Güven: %{round(d.get('confidence', 0)*100, 1)}, Yüzey Oranı: %{d.get('surface_ratio_pct')}, Şiddet: {d.get('severity')}\n"

        if not defects_str:
            defects_str = "Herhangi bir belirli kusur tespit edilmedi (Temiz yüzey)."

        return f"""Aşağıda bir ahşap/kereste yüzeyine ait yapay zeka tespit sisteminden elde edilen analiz verileri bulunmaktadır:

Kalite Sınıfı: {analysis_result.get('grade')}
Sınıf Açıklaması: {analysis_result.get('grade_explanation')}
Toplam Kusur Sayısı: {analysis_result.get('total_defects')}
Kusurlu Alan Oranı: %{analysis_result.get('total_defect_ratio_pct')}

Kusur Detayları:
{defects_str}

Talimatlar:
- Yalnızca yukarıda verilen tespit verilerine dayan.
- Yeni sayısal ölçüm veya veri uydurma.
- Emin olmadığın değerlendirmeleri belirt.
- Yanıtı tam olarak şu sabit Markdown başlıklarıyla Türkçe ver:

### 1. Genel Değerlendirme
### 2. Kusurların Nedeni ve Etkileri
### 3. Kullanım Alanı Uygunluğu
### 4. Üretim Hattı İyileştirme Önerileri
### 5. Tespit Modelinin Olası Yanılma Noktaları"""

    def generate_ai_analysis(self, analysis_result: dict) -> dict:
        if not self.api_key:
            return {
                "success": False,
                "error": "Google Gemini API anahtarı (GEMINI_API_KEY) .env dosyasında tanımlanmamış."
            }

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = self._build_prompt(analysis_result)

            model_name = self.config.get("model", "gemini-3.8-flash")

            # 503 Yoğunluk hatalarına karşı 3 defa otomatik yeniden deneme (Retry)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    text_content = response.text if response.text else ""
                    return {
                        "success": True,
                        "report_text": text_content
                    }
                except Exception as req_err:
                    err_msg = str(req_err)
                    # Geçici yoğunluk (503 / UNAVAILABLE) durumunda bekle ve tekrar dene
                    if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise req_err

        except Exception as e:
            logger.exception("AI Analiz üretilirken hata oluştu:")
            return {
                "success": False,
                "error": f"Gemini API servisine bağlanılamadı (Yoğunluk nedeniyle geçici hata): {str(e)}"
            }