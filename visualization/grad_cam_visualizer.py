import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().numpy()

def visualize_gradcam(image_path, model_path="resnet_model.pth"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    target_layer = model.layer4[1].conv2

    cam_generator = GradCAM(model, target_layer)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    cam = cam_generator.generate(input_tensor)

    cam = cv2.resize(cam, (224, 224))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image.resize((224, 224)))
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.show()

# Example usage
# visualize_gradcam("sample_image.jpg")
