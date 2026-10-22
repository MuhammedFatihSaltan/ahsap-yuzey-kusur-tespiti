import sys
import uuid
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.detector import YOLODetector
from ahsapgoz.anomaly import AnomalyDetector
from ahsapgoz.analysis import DefectAnalyzer
from ahsapgoz.report import PDFReportGenerator

app = FastAPI(title="AhşapGöz Web Uygulaması")

# Statik Dosyalar ve Şablonlar
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# Modellerin ve Servislerin Yüklenmesi (Eşik 0.03 olarak ayarlandı)
yolo_detector = YOLODetector(conf_threshold=0.03)
anomaly_detector = AnomalyDetector()
analyzer = DefectAnalyzer()
pdf_generator = PDFReportGenerator()

# Bellekte son analiz verisini saklama
LATEST_ANALYSIS = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    global LATEST_ANALYSIS
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return templates.TemplateResponse(request=request, name="index.html", context={"error": "Geçersiz görsel dosyası!"})

    file_id = str(uuid.uuid4())[:8]
    
    # 1. YOLO Tespiti
    yolo_res = yolo_detector.predict(image)
    
    # 2. Anomali Isı Haritası
    anomaly_res = anomaly_detector.compute_heatmap(image)

    # 3. Kalite Sınıflandırma (Anomali verisi de aktarılıyor)
    surface_area = image.shape[0] * image.shape[1]
    analysis_res = analyzer.calculate_quality_grade(yolo_res["detections"], surface_area, anomaly_res)

    # İşlenmiş Görselleri Kaydetme
    uploads_dir = BASE_DIR / "outputs" / "uploads"
    results_dir = BASE_DIR / "outputs" / "results"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    ann_path = results_dir / f"annotated_{file_id}.jpg"
    heat_path = results_dir / f"heatmap_{file_id}.jpg"

    cv2.imwrite(str(ann_path), yolo_res["annotated_image"])
    cv2.imwrite(str(heat_path), anomaly_res["overlay"])

    LATEST_ANALYSIS = {
        "id": file_id,
        "result": analysis_res,
        "ann_path": str(ann_path),
        "heat_path": str(heat_path)
    }

    context = {
        "result": analysis_res,
        "annotated_image_url": f"/outputs/results/annotated_{file_id}.jpg",
        "heatmap_image_url": f"/outputs/results/heatmap_{file_id}.jpg",
        "file_id": file_id
    }

    return templates.TemplateResponse(request=request, name="result.html", context=context)

@app.get("/report/{file_id}.pdf")
async def download_pdf_report(file_id: str):
    reports_dir = BASE_DIR / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(reports_dir / f"rapor_{file_id}.pdf")

    if LATEST_ANALYSIS.get("id") == file_id:
        pdf_generator.generate_report(
            LATEST_ANALYSIS["result"],
            LATEST_ANALYSIS["ann_path"],
            LATEST_ANALYSIS["heat_path"],
            pdf_path
        )
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"AhsapGoz_Rapor_{file_id}.pdf")
    
    return {"error": "Rapor bulunamadı veya süresi doldu."}