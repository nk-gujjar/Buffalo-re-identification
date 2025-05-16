
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import dotenv
import os
# Load environment variables from .env file
dotenv.load_dotenv()

# from huggingface_hub import login

# login(token='huggingface_token')



# Define your model class (same as training)
class EfficientNetForAnimalRecognition(nn.Module):
    def __init__(self, num_classes=1072):
        super().__init__()
        # --- Use timm model instead of torchvision model ---
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove the classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Not needed for timm but keeping as you had it
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        features = self.efficientnet.forward_features(x)
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        return self.classifier(features)

    def extract_features(self, x):
        features = self.efficientnet.forward_features(x)
        features = self.global_pool(features)
        return torch.flatten(features, 1)

# Image preprocessing (must match training)
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Load model and weights
def load_model(path, device):
    model = EfficientNetForAnimalRecognition()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Extract normalized feature vector
def get_normalized_features(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.extract_features(image_tensor)
        features = features / features.norm(dim=1, keepdim=True)
    return features.cpu().numpy()
