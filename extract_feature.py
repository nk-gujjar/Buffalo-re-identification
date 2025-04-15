# # extract_feature.py

# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from model import load_model  # import from model.py

# # Preprocess image to match EfficientNet-B3 input
# def preprocess_image(img_path):
#     transform = transforms.Compose([
#         transforms.Resize((300, 300)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     image = Image.open(img_path).convert("RGB")
#     return transform(image).unsqueeze(0)  # Add batch dimension

# # Extract normalized feature vector
# def get_normalized_features(model, image_tensor, device):
#     image_tensor = image_tensor.to(device)
#     with torch.no_grad():
#         features = model.extract_features(image_tensor)
#         features = features / features.norm(dim=1, keepdim=True)
#     return features.cpu().numpy()

# # Example usage
# if __name__ == "__main__":
#     from sklearn.metrics.pairwise import cosine_similarity

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model("best_model.pth", device)

#     img1 = preprocess_image("sample_img/2_a.jpg")
#     img2 = preprocess_image("sample_img/2_b.jpg")
    
#     feat1 = get_normalized_features(model, img1, device)
#     feat2 = get_normalized_features(model, img2, device)
#     print("Feature Vector 1:", feat1)
#     print("Feature Vector 2:", feat2)

#     similarity = cosine_similarity(feat1, feat2)
#     print("Cosine Similarity:", similarity[0][0])


import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import load_model  # import from model.py
import io

# Preprocess image to match EfficientNet-B3 input
def preprocess_image(img_data):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Handle both file paths and binary image data
    if isinstance(img_data, str):
        image = Image.open(img_data).convert("RGB")
    else:
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
    return transform(image).unsqueeze(0)  # Add batch dimension

# Extract normalized feature vector
def get_normalized_features(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        features = model.extract_features(image_tensor)
        features = features / features.norm(dim=1, keepdim=True)
    return features.cpu().numpy()

# Main function to extract features from image data
def extract_features(img_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("best_model.pth", device)
    img_tensor = preprocess_image(img_data)
    feature_vector = get_normalized_features(model, img_tensor, device)
    return feature_vector.flatten().tolist()  # Convert to list for MongoDB storage

# Example usage
if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("best_model.pth", device)
    img1 = preprocess_image("sample_img/2_a.jpg")
    img2 = preprocess_image("sample_img/2_b.jpg")
    
    feat1 = get_normalized_features(model, img1, device)
    feat2 = get_normalized_features(model, img2, device)
    print("Feature Vector 1:", feat1)
    print("Feature Vector 2:", feat2)
    similarity = cosine_similarity(feat1, feat2)
    print("Cosine Similarity:", similarity[0][0])