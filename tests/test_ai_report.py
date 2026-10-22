import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.ai_report import AIReportGenerator

def test_ai_report_generator_mock():
    generator = AIReportGenerator()
    generator.api_key = "gsk_dummy_key"

    mock_analysis = {
        "grade": "C Sınıfı",
        "grade_explanation": "Test",
        "total_defects": 1,
        "total_defect_ratio_pct": 3.5,
        "defects": [
            {
                "confidence": 0.85,
                "surface_ratio_pct": 3.5,
                "severity": "Orta",
                "details": {"ad": "Çatlak", "onerilen_islem": "Dolgu"}
            }
        ]
    }

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "### 1. Genel Değerlendirme\nYüzeyde çatlak mevcut."
    mock_response.choices = [mock_choice]

    with patch("groq.Groq") as mock_groq:
        instance = mock_groq.return_value
        instance.chat.completions.create.return_value = mock_response

        res = generator.generate_ai_analysis(mock_analysis)

        assert res["success"] is True
        assert "Genel Değerlendirme" in res["report_text"]