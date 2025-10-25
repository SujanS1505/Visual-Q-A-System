from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# Pick lightweight BLIP-2 model (available on Hugging Face)
# Options: "Salesforce/blip2-opt-2.7b" (lighter) or "Salesforce/blip2-flan-t5-xl" (heavier, more accurate)
model_name = "Salesforce/blip2-opt-2.7b"

# Load processor + model
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

def get_blip2_answer(image: Image.Image, question: str) -> str:
    """Generate answer using BLIP-2 model."""
    inputs = processor(image, question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
