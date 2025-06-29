import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import sys
import io
import torch
from torchvision import models
import timm
from tensorflow.keras.preprocessing import image as keras_image
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import urllib.parse

import timm
import torch.nn as nn
import torch
from torch.optim import AdamW
# Thiết lập stdout để in UTF-8 (chỉ cần trên Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Cấu hình đường dẫn
model_path = r"E:\similarity_image\models\convnextv2\convnext_v2_best_params_aug2_final.pth"
input_dir = r"E:\LuanVan\data\test-others"
output_dir = r"E:\LuanVan\data\phanloai_ketqua1"
img_size = (224, 224)

def build_convnext_v2_aug_model(num_classes, lr=0.000188, dropout_rate=0.204782, dense_units=576,
                l2_reg=1.792511e-07 , fine_tune_all=False, unfreeze_from_stage=2):

    # Tải mô hình ConvNeXt v2 với pretrained weights
    model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True, num_classes=0)
    in_features = model.head.in_features

    # Đóng băng toàn bộ nếu không fine-tune tất cả
    for param in model.parameters():
        param.requires_grad = False

    # Mở các stage từ vị trí chỉ định (unfreeze từ stage 2 trở đi)
    for i, stage in enumerate(model.stages):
        if i >= unfreeze_from_stage:
            for param in stage.parameters():
                param.requires_grad = True

    # Head mới với các tham số từ trial
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features, dense_units),
        nn.GELU(),  
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, num_classes)
    )

    return model


def load_convnextv2_model(model_path, num_classes):
    model = build_convnext_v2_aug_model(num_classes=num_classes)
    
    # Load trọng số
    state_dict = torch.load(model_path, map_location='cpu')

    # Trường hợp lưu theo dạng full dict có 'model_state_dict'
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
    else:
        model_state_dict = state_dict  # Trường hợp load trực tiếp state_dict

    if 'head.5.weight' in model_state_dict:
        print("✔ Found key: head.5.weight, Shape:", model_state_dict['head.5.weight'].shape)
    else:
        print("⚠ Không tìm thấy key 'head.5.weight'. Có thể model đã đổi cấu trúc head.")

    # Load trọng số vào mô hình
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

    # Thông báo nếu có keys không khớp
    if missing_keys:
        print("⚠ Missing keys:", missing_keys)
    if unexpected_keys:
        print("⚠ Unexpected keys:", unexpected_keys)

    if missing_keys or unexpected_keys:
        print("⚠ Cảnh báo: Có key không khớp, kiểm tra lại số lớp hoặc cấu trúc head của mô hình!")

    model.eval()
    return model

preprocess_torch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load model
model = load_convnextv2_model(model_path, 11)

# Lấy tên lớp từ thư mục huấn luyện cũ
ref_dir = r"E:\LuanVan\data\split-raw\train"
class_names = sorted([d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d))])
print("🔍 Lớp phân loại:", class_names)

# Tạo thư mục đích cho từng lớp
for class_name in class_names:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# Duyệt qua các ảnh trong thư mục cần phân loại
for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, fname)

        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess_torch(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            preds = torch.softmax(preds, dim=1)
        class_id = torch.argmax(preds[0]).item()
        confidence = preds[0][class_id].item()
        predicted_class = class_names[class_id]

        # Decode tên file từ URL encoding sang tên bình thường
        decoded_fname = urllib.parse.unquote(fname)

        # Tạo thư mục đích nếu chưa có
        dest_dir = os.path.join(output_dir, predicted_class)
        os.makedirs(dest_dir, exist_ok=True)

        # Tạo đường dẫn đích
        dest_path = os.path.join(dest_dir, decoded_fname)

        # Di chuyển ảnh vào thư mục lớp tương ứng
        
        shutil.copy(img_path, dest_path)
        print(f"📂 {fname} ➜ {predicted_class}")

print("✅ Phân loại hoàn tất!")
