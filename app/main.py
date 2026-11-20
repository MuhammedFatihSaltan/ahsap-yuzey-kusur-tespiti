import sys
import uuid
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse, StreamingResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from ahsapgoz.detector import YOLODetector
from ahsapgoz.anomaly import AnomalyDetector
from ahsapgoz.analysis import DefectAnalyzer
from ahsapgoz.report import PDFReportGenerator
from ahsapgoz.ai_report import AIReportGenerator

app = FastAPI(title="AhşapGöz Web Uygulaması")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

yolo_detector = YOLODetector(conf_threshold=0.03)
anomaly_detector = AnomalyDetector()
analyzer = DefectAnalyzer()
pdf_generator = PDFReportGenerator()
ai_generator = AIReportGenerator()

LATEST_ANALYSIS = {}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

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
    
    yolo_res = yolo_detector.predict(image)
    anomaly_res = anomaly_detector.compute_heatmap(image)

    surface_area = image.shape[0] * image.shape[1]
    analysis_res = analyzer.calculate_quality_grade(yolo_res["detections"], surface_area, anomaly_res)

    uploads_dir = BASE_DIR / "outputs" / "uploads"
    results_dir = BASE_DIR / "outputs" / "results"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    ann_path = results_dir / f"annotated_{file_id}.jpg"
    heat_path = results_dir / f"heatmap_{file_id}.jpg"

    final_ann_img = image.copy()
    for det in analysis_res.get("filtered_detections", []):
        x, y, w, h = [int(v) for v in det["bbox"]]
        label_text = f"{det['label']} %{round(det['confidence'] * 100, 1)}"
        color = (0, 200, 0) if det["label"] != "anomali_kusur" else (0, 215, 255)
        cv2.rectangle(final_ann_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_ann_img, label_text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite(str(ann_path), final_ann_img)
    cv2.imwrite(str(heat_path), anomaly_res["overlay"])

    LATEST_ANALYSIS = {
        "id": file_id,
        "result": analysis_res,
        "ann_path": str(ann_path),
        "heat_path": str(heat_path),
        "ai_report": None
    }

    context = {
        "result": analysis_res,
        "annotated_image_url": f"/outputs/results/annotated_{file_id}.jpg",
        "heatmap_image_url": f"/outputs/results/heatmap_{file_id}.jpg",
        "file_id": file_id
    }

    return templates.TemplateResponse(request=request, name="result.html", context=context)

@app.post("/ai-analyze-stream/{file_id}")
def run_ai_analysis_stream(file_id: str):
    if LATEST_ANALYSIS.get("id") != file_id:
        return JSONResponse(status_code=404, content={"error": "Analiz verisi bulunamadı."})

    if LATEST_ANALYSIS.get("ai_report"):
        def cached_gen():
            yield LATEST_ANALYSIS["ai_report"]
        return StreamingResponse(cached_gen(), media_type="text/plain; charset=utf-8")

    def event_generator():
        full_chunks = []
        for chunk in ai_generator.generate_ai_analysis_stream(LATEST_ANALYSIS["result"]):
            full_chunks.append(chunk)
            yield chunk
        final_text = "".join(full_chunks)
        # Hata veya uyarı mesajlarının önbelleğe kaydedilmesini engelle
        if final_text.strip() and not final_text.strip().startswith(("[Sistem Uyarısı", "HATA:")):
            LATEST_ANALYSIS["ai_report"] = final_text

    return StreamingResponse(event_generator(), media_type="text/plain; charset=utf-8")

@app.post("/ai-analyze/{file_id}")
def run_ai_analysis(file_id: str):
    if LATEST_ANALYSIS.get("id") != file_id:
        return JSONResponse(status_code=404, content={"success": False, "error": "Analiz verisi bulunamadı."})

    if LATEST_ANALYSIS.get("ai_report"):
        return {"success": True, "report_text": LATEST_ANALYSIS["ai_report"]}

    res = ai_generator.generate_ai_analysis(LATEST_ANALYSIS["result"])
    if res.get("success"):
        LATEST_ANALYSIS["ai_report"] = res.get("report_text")
        return {"success": True, "report_text": res.get("report_text")}
    else:
        return JSONResponse(status_code=500, content={"success": False, "error": res.get("error")})

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
            pdf_path,
            ai_text=LATEST_ANALYSIS.get("ai_report")
        )
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"AhsapGoz_Rapor_{file_id}.pdf")
    
    return {"error": "Rapor bulunamadı veya süresi doldu."}