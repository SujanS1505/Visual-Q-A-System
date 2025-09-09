from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# Load model + processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load a sample image (put noodles.jpg in the same folder)
image = Image.open("noodles.jpg")

# Ask a question
question = "What is the person doing?"

# Preprocess inputs
inputs = processor(image, question, return_tensors="pt")

# Run model
outputs = model(**inputs)
logits = outputs.logits
idx = logits.argmax(-1).item()
answer = model.config.id2label[idx]

print("Question:", question)
print("Answer:", answer)
