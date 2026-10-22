import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class PDFReportGenerator:
    """AhşapGöz Profesyonel PDF Rapor Motoru."""

    def __init__(self):
        self._register_fonts()
        self.styles = getSampleStyleSheet()

    def _register_fonts(self):
        try:
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            font_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont("Arial-TR", font_path))
                pdfmetrics.registerFont(TTFont("Arial-TR-Bold", font_bold_path if os.path.exists(font_bold_path) else font_path))
                self.font_regular = "Arial-TR"
                self.font_bold = "Arial-TR-Bold"
            else:
                self.font_regular = "Helvetica"
                self.font_bold = "Helvetica-Bold"
        except Exception:
            self.font_regular = "Helvetica"
            self.font_bold = "Helvetica-Bold"

    def _get_grade_color(self, grade: str) -> str:
        if "Red" in grade:
            return "#D32F2F"
        elif "C" in grade:
            return "#E65100"
        return "#2E7D32"

    def generate_report(self, analysis_result: dict, image_path: str, heatmap_path: str, output_pdf_path: str, ai_text: str = None) -> str:
        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        
        story = []

        title_style = ParagraphStyle(
            "TitleStyle",
            fontName=self.font_bold,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#5D4037"),
            alignment=1,
            spaceAfter=12
        )

        normal_style = ParagraphStyle(
            "NormalStyle",
            fontName=self.font_regular,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2C221E")
        )

        bold_style = ParagraphStyle(
            "BoldStyle",
            fontName=self.font_bold,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2C221E")
        )

        header_table_style = ParagraphStyle(
            "HeaderTableStyle",
            fontName=self.font_bold,
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#FFFFFF")
        )

        # 1. Başlık
        story.append(Paragraph("AhşapGöz - Ahşap Yüzey Kalite Kontrol Raporu", title_style))
        story.append(Spacer(1, 8))

        # 2. Kalite Özeti Kutusu
        grade = analysis_result.get('grade', 'N/A')
        grade_color = self._get_grade_color(grade)
        
        summary_text = f"<b>Kalite Sınıfı:</b> <font color='{grade_color}'><b>{grade}</b></font><br/>" \
                       f"<b>Açıklama:</b> {analysis_result.get('grade_explanation', '')}<br/>" \
                       f"<b>Toplam Kusur Sayısı:</b> {analysis_result.get('total_defects', 0)} | " \
                       f"<b>Kusurlu Alan Oranı:</b> %{analysis_result.get('total_defect_ratio_pct', 0.0)}"
        story.append(Paragraph(summary_text, normal_style))
        story.append(Spacer(1, 12))

        # 3. Görseller
        if os.path.exists(image_path) and os.path.exists(heatmap_path):
            img1 = Image(image_path, width=220, height=160)
            img2 = Image(heatmap_path, width=220, height=160)
            img_table = Table([[img1, img2]], colWidths=[240, 240])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 12))

        # 4. Tablo
        table_data = [[
            Paragraph("Kusur Türü", header_table_style),
            Paragraph("Güven", header_table_style),
            Paragraph("Yüzey %", header_table_style),
            Paragraph("Şiddet", header_table_style),
            Paragraph("Önerilen İşlem", header_table_style)
        ]]

        for defect in analysis_result.get("defects", []):
            details = defect.get("details", {})
            label_name = details.get("ad", defect.get("label", "Yüzey Kusuru"))
            table_data.append([
                Paragraph(f"<b>{label_name}</b>", bold_style),
                f"%{round(defect.get('confidence', 0.0) * 100, 1)}",
                f"%{defect.get('surface_ratio_pct', 0.0)}",
                defect.get("severity", "Yüksek"),
                Paragraph(details.get("onerilen_islem", "Uzman kontrolü tavsiye edilir."), normal_style)
            ])

        if len(table_data) == 1:
            table_data.append([
                Paragraph("Temiz Yüzey", normal_style),
                "-", "-", "-",
                Paragraph("İşlem gerekmiyor.", normal_style)
            ])

        defect_table = Table(table_data, colWidths=[100, 45, 45, 45, 245])
        defect_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#5D4037")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D7CCC8")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(defect_table)
        story.append(Spacer(1, 12))

        # 5. Opsiyonel AI Analiz Bölümü
        if ai_text:
            ai_title_style = ParagraphStyle(
                "AITitleStyle",
                fontName=self.font_bold,
                fontSize=12,
                leading=15,
                textColor=colors.HexColor("#D97706"),
                spaceAfter=6
            )
            ai_note_style = ParagraphStyle(
                "AINoteStyle",
                fontName=self.font_regular,
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#6B7280"),
                spaceAfter=8
            )

            story.append(Paragraph("<b>Yapay Zekâ Detaylı Analiz Değerlendirmesi</b>", ai_title_style))
            story.append(Paragraph("<i>* Bu bölüm yapay zekâ tarafından üretilmiştir, uzman onayı gerektirir.</i>", ai_note_style))

            formatted_ai = ai_text.replace("\n", "<br/>").replace("### ", "<br/><b>").replace("<b>1.", "<b>1.").replace("<b>2.", "<b>2.")
            story.append(Paragraph(formatted_ai, normal_style))
            story.append(Spacer(1, 10))

        # 6. Sorumluluk Reddi
        disclaimer = "<i>Not: Bu rapor otomatik yapay zekâ analiz sistemi tarafından üretilmiştir. " \
                     "Nihai kalite kararları için uzman denetimi tavsiye edilir.</i>"
        story.append(Paragraph(disclaimer, normal_style))

        doc.build(story)
        return output_pdf_path