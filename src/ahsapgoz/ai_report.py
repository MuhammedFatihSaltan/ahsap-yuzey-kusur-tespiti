import os
import json
import logging
import urllib.request
from pathlib import Path
from collections import Counter
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)


class AIReportGenerator:
    """Kapsamlı ve hızlı ahşap yüzey analiz raporu üretici (Çoklu Model Destekli)."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"

        self.config = self._load_config(config_path)
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")

        self.gemini_client = (
            genai.Client(
                api_key=self.gemini_key,
                http_options=types.HttpOptions(timeout=15000)
            )
            if self.gemini_key else None
        )
        self.available_gemini_models = self._discover_gemini_models()

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
            "max_tokens": 1200,
        }

    @staticmethod
    def _is_valid_text_model(model_name: str) -> bool:
        lower_name = model_name.lower()
        skip_keywords = ("embed", "imagen", "veo", "aqa", "tts", "audio")
        preferred_keywords = ("flash", "pro", "gemma-3-27b", "gemma")
        if any(skip in lower_name for skip in skip_keywords):
            return False
        return any(kw in lower_name for kw in preferred_keywords)

    @staticmethod
    def _model_priority(model_name: str) -> int:
        ln = model_name.lower()
        if "3.8-flash" in ln:
            return 1
        if "flash" in ln:
            return 2
        if "pro" in ln:
            return 3
        if "27b" in ln:
            return 4
        return 5

    def _discover_gemini_models(self) -> list:
        """Hesaptaki aktif tüm metin üretim modellerini öncelik sırasına göre listeler."""
        if not self.gemini_client:
            return []

        discovered = []
        try:
            for m in self.gemini_client.models.list():
                name = getattr(m, "name", "").removeprefix("models/")
                if self._is_valid_text_model(name):
                    discovered.append(name)
        except Exception as e:
            logger.warning("Gemini model listesi alınamadı: %s", e)

        sorted_models = sorted(dict.fromkeys(discovered), key=self._model_priority)
        return sorted_models or ["gemini-3.8-flash"]

    def _build_prompt(self, analysis_result: dict) -> str:
        defects = analysis_result.get("defects", [])
        total_defects = analysis_result.get("total_defects", len(defects))

        if not defects:
            defects_summary = "Kusur tespit edilmedi (Temiz yüzey)."
        else:
            type_counts = Counter(
                d.get("details", {}).get("ad", d.get("label", "Bilinmeyen")) for d in defects
            )
            summary_parts = [f"{k}: {v} adet" for k, v in type_counts.items()]
            defects_summary = "Kusur Türü Dağılımı: " + ", ".join(summary_parts) + "\nDetaylı Kusur Listesi:\n"

            for idx, d in enumerate(defects[:8], 1):
                details = d.get("details", {})
                conf_pct = round(d.get("confidence", 0.0) * 100, 1)
                defects_summary += (
                    f"{idx}. {details.get('ad', d.get('label'))} | "
                    f"Güven Skoru: %{conf_pct} | Yüzey Oranı: %{d.get('surface_ratio_pct')} | "
                    f"Şiddet: {d.get('severity')} | Mevcut Öneri: {details.get('onerilen_islem', '-')}\n"
                )

        return f"""Sen kıdemli bir ahşap malzeme mühendisi ve endüstriyel kalite kontrol uzmanısın. Aşağıdaki görüntü işleme ve yapay zekâ tespit verilerini analiz ederek kapsamlı, teknik ve yol gösterici bir kalite kontrol raporu hazırla.

Analiz Verileri:
- Kalite Sınıfı: {analysis_result.get('grade')}
- Sistem Açıklaması: {analysis_result.get('grade_explanation')}
- Toplam Tespit Edilen Kusur Sayısı: {total_defects}
- Toplam Kusurlu Yüzey Alanı Oranı: %{analysis_result.get('total_defect_ratio_pct')}
{defects_summary}

Yazım Kuralları:
- Yalnızca verilen verilere ve sayısal değerlere dayan; veride olmayan sayı uydurma.
- Akıcı, profesyonel, teknik derinliği olan ve açıklayıcı bir Türkçe kullan (yaklaşık 300-400 kelime).
- Önemli kavramları ve sayısal bulguları **kalın** yazarak vurgula.
- Tam olarak şu 5 Markdown başlığını kullan ve her başlığın altını doyurucu şekilde doldur:

### 1. Genel Değerlendirme
### 2. Kusurların Olası Kök Nedenleri ve Yapısal/Estetik Etkileri
### 3. Kullanım Alanı Uygunluğu ve Sınıflandırma Yorumu
### 4. Üretim Hattı ve İşleme İyileştirme Önerileri
### 5. Tespit Modelinin Güvenilirliği ve Olası Yanılma Noktaları"""

    def _stream_from_groq(self, prompt: str):
        """Groq API üzerinden Llama 3.3 70B ile ultra hızlı canlı akış üretir."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1200,
            "stream": True
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
                "User-Agent": "AhsapGoz/1.0"
            }
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    yield delta

    def _stream_single_gemini_model(self, model_name: str, prompt: str):
        """Tek bir Gemini modelinden akış verisi çeker."""
        cfg_kwargs = {
            "max_output_tokens": int(self.config.get("max_tokens", 1200)),
            "temperature": 0.35,
        }
        if "gemini-3" in model_name or "gemini-2.5" in model_name:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        response_stream = self.gemini_client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs)
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def generate_ai_analysis_stream(self, analysis_result: dict):
        """Önce Groq (varsa) veya aktif Gemini modellerini tarayarak kesintisiz canlı analiz üretir."""
        prompt = self._build_prompt(analysis_result)

        if self.groq_key:
            try:
                chunks = list(self._stream_from_groq(prompt))
                if chunks:
                    yield from chunks
                    return
            except Exception as e:
                logger.warning("Groq stream başarısız oldu, Gemini modellerine geçiliyor: %s", e)

        if not self.gemini_client:
            yield "HATA: .env dosyasında GEMINI_API_KEY veya GROQ_API_KEY tanımlı değil."
            return

        last_error = None
        for model_name in self.available_gemini_models:
            try:
                has_yielded = False
                for chunk in self._stream_single_gemini_model(model_name, prompt):
                    has_yielded = True
                    yield chunk
                if has_yielded:
                    return
            except Exception as e:
                last_error = e
                logger.warning("Model (%s) atlandı (%s), sonraki modele geçiliyor...", model_name, str(e))

        yield f"[Sistem Uyarısı: Tüm AI modelleri şu an yoğunluk nedeniyle yanıt veremiyor. ({str(last_error)})]"

    def generate_ai_analysis(self, analysis_result: dict) -> dict:
        full_text = "".join(self.generate_ai_analysis_stream(analysis_result))
        if full_text.strip() and not full_text.strip().startswith(("[Sistem Uyarısı", "HATA:")):
            return {"success": True, "report_text": full_text}
        return {"success": False, "error": full_text}