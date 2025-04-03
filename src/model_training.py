import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch import nn, optim
from tqdm import tqdm

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load sample dataset (can be replaced with pneumonia or skin cancer dataset)
train_dataset = torchvision.datasets.FakeData(transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load pre-trained ResNet model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "resnet_model.pth")
print("Training completed and model saved.")
