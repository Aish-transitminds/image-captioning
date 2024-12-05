# image-captioning
pip install transformers pillow torch [pip install is most required]
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess image
image = Image.open("/kri.jpg")  # Replace with your image path
inputs = processor(image, return_tensors="pt")

# Generate caption
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print("Caption:", caption)
