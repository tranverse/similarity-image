import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import os

# Tải mô hình đã lưu
convnext_v2 = models.convnext_v2(pretrained=False)
convnext_v2.load_state_dict(torch.load('convnext_v2.pth'))
convnext_v2.eval()

# Tiền xử lý hình ảnh
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)
    return img

# Hàm trích xuất đặc trưng và SPoC pooling
def extract_spoc_features_convnext_v2(img_path):
    img = preprocess_image(img_path)
    
    with torch.no_grad():
        # Trích xuất đặc trưng từ mô hình
        features = convnext_v2(img)
        
    # Sum pooling: Tổng hợp đặc trưng theo chiều W và H
    pooled_features = torch.sum(features, dim=(2, 3))  # Tổng hợp theo chiều W và H
    
    # Chuyển đổi tensor thành mảng numpy
    pooled_features = pooled_features.cpu().numpy()
    
    # Chuẩn hóa L2
    normalized_features = pooled_features / np.linalg.norm(pooled_features, axis=1, keepdims=True)
    
    # Nén bằng PCA
    pca = PCA(n_components=256)
    pca_features = pca.fit_transform(normalized_features)
    
    return pca_features

