from fastapi import FastAPI, UploadFile, Form
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import io

app = FastAPI()

# Load model + processor once globally
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.post("/ask")
async def ask_image_question(file: UploadFile, question: str = Form(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)

    # Get answer
    idx = outputs.logits.argmax(-1).item()
    answer = model.config.id2label[idx]

    return {"question": question, "answer": answer}
