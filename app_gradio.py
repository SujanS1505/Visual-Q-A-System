import gradio as gr
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import ssl
import os

# Disable SSL verification (use cautiously)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

# Load model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def vqa(image, question):
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)
    idx = outputs.logits.argmax(-1).item()
    return model.config.id2label[idx]

# Launch UI
gr.Interface(
    fn=vqa,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Ask a question")],
    outputs="text"
).launch()
