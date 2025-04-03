from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import io

app = FastAPI()

# Load model
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        class_name = "Pneumonia" if predicted.item() == 1 else "Normal"
        return JSONResponse(content={"prediction": class_name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
