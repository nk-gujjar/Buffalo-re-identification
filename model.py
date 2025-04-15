# model.py

import torch
import torch.nn as nn
import timm

# Define model class
class EfficientNetForAnimalRecognition(nn.Module):
    def __init__(self, num_classes=1072):
        super().__init__()
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
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

# Function to load the model with weights
def load_model(path, device):
    model = EfficientNetForAnimalRecognition()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
