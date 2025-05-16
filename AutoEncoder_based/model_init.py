import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

# #First define the Autoencoder class exactly as used during training
# class Autoencoder(nn.Module):
#     def __init__(self, latent_dim=128):
#         super(Autoencoder, self).__init__()
#         # Encoder (EfficientNet)
#         self.encoder = models.efficientnet_b0(weights=None)
#         self.encoder.classifier = nn.Identity()
        
#         self.encoder_fc = nn.Sequential(
#             nn.Linear(1280, 512),
#             nn.ReLU(),
#             nn.Linear(512, latent_dim),
#             nn.ReLU()
#         )
        
#         # Decoder (matches your training architecture)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 7*7*256),
#             nn.ReLU(),
#             nn.Unflatten(1, (256, 7, 7)),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         features = self.encoder(x)
#         encoded = self.encoder_fc(features)
#         return self.decoder(encoded)
    
#     def encode(self, x):
#         with torch.no_grad():
#             features = self.encoder(x)
#             return self.encoder_fc(features)

# # Feature extractor wrapper
# class MuzzleFeatureExtractor(nn.Module):
#     def __init__(self, model_path):
#         super().__init__()
#         # Load full autoencoder
#         full_model = Autoencoder()
#         full_model.load_state_dict(torch.load(model_path))
        
#         # Extract encoder components
#         self.encoder = full_model.encoder
#         self.encoder_fc = full_model.encoder_fc
        
#         # Freeze all parameters
#         for param in self.parameters():
#             param.requires_grad = False
            
#         # Feature normalization
#         self.norm = nn.LayerNorm(128)
        
#     def forward(self, x):
#         features = self.encoder(x)
#         features = self.encoder_fc(features)
#         return self.norm(features)

# # Image preprocessing
# def preprocess_image(img_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                            std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(img_path).convert("RGB")
#     return transform(image).unsqueeze(0)

# # Model loading
# def load_autoencoder(model_path, device):
#     model = Autoencoder()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()
#     return model



import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = models.efficientnet_b0(weights=None)
        self.encoder.classifier = nn.Identity()

        self.encoder_fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 7*7*256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        encoded = self.encoder_fc(features)
        return self.decoder(encoded)

    def encode(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            return self.encoder_fc(features)

def load_model(checkpoint_path, device, latent_dim=128):
    model = Autoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, 3, 224, 224)








# # Feature extraction with enhanced normalization
# def get_enhanced_features(model, image_tensor, device):
#     image_tensor = image_tensor.to(device)
#     with torch.no_grad():
#         features = model(image_tensor)
#         # Double normalization
#         features = features / features.norm(dim=1, keepdim=True)
#         features = 0.5 * (features + 1)  # Scale to [0,1] range
#     return features.cpu().numpy()