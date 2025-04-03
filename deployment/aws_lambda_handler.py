import json
import base64
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import io

# Load model
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("resnet_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def lambda_handler(event, context):
    try:
        body = event.get('body')
        if event.get("isBase64Encoded"):
            image_data = base64.b64decode(body)
        else:
            return {"statusCode": 400, "body": "Invalid image encoding."}

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        class_name = "Pneumonia" if predicted.item() == 1 else "Normal"
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": class_name})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
