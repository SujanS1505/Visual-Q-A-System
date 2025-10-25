from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from utils.blip2_model import get_blip2_answer

app = FastAPI()

@app.get("/")
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/ask")
async def ask_question(file: UploadFile, question: str = Form(...)):
    image = Image.open(io.BytesIO(await file.read()))
    answer = get_blip2_answer(image, question)
    return {"answer": answer}
