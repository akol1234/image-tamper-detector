from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from torchvision import models, transforms
from PIL import Image, ImageChops
import shutil
import os
import uuid

app = FastAPI()

templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 1)

checkpoint = torch.load(
    "/content/ela_mobilenet_tamper_detector.pthh",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
threshold = checkpoint.get("threshold", 0.3)

model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def compute_ela(path, quality=90):
    img = Image.open(path).convert("RGB")
    temp_path = f"{path}_temp.jpg"
    img.save(temp_path, "JPEG", quality=quality)
    compressed = Image.open(temp_path)
    ela = ImageChops.difference(img, compressed)
    ela = ela.point(lambda x: x * 10)
    os.remove(temp_path)
    return ela


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    unique_name = f"{uuid.uuid4()}.jpg"
    upload_path = f"uploads/{unique_name}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ela = compute_ela(upload_path)
    tensor = tf(ela).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    os.remove(upload_path)

    result = "tampered" if prob > threshold else "original"

    return {
        "prediction": result,
        "confidence": float(prob)
    }