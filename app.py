import io
import os
import ssl
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor
import torch

# Disable SSL verification for model downloads (use cautiously in production)
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Visual QA System", version="1.0.0")

# Enable CORS for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files so / serves the HTML frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load model once at startup
torch.set_grad_enabled(False)
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/v1/vqa")
async def vqa_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    inputs = processor(pil_image, question, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        idx = outputs.logits.argmax(-1).item()
        answer = model.config.id2label[idx]

    return {"answer": answer}
