import os
import numpy as np
import faiss
import joblib
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA
import django
import sys

# Setup Django
sys.path.append('E:/similarity_image')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.similarity_image.settings')
django.setup()

from backend.api.models import Research, Feature

# Model and paths
MODEL_PATH = r"E:\convnext_v2_best_params_aug_final_11_7.pth"
INDEX_DIR = r'E:/similarity_image/extract_features'
IMG_FOLDER = r'E:\similarity_image\dataset'
MODEL_TYPE = 'convnext_v2_aug'
PCA_COMPONENTS = 256

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

import timm
import torch.nn as nn
import torch
from torch.optim import AdamW

def build_model(num_classes, lr=0.000466, 
                l2_reg=0.000017  , fine_tune_all=False, unfreeze_from_stage=3):

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
        nn.GELU(),  
        nn.Linear(in_features, 11) # Linear classifier
    )

    return model


# Load ConvNeXt V2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(num_classes=len(class_names))
checkpoint = torch.load(MODEL_PATH, map_location=device)
# print(checkpoint['model_state_dict'].keys())

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
model.to(device)


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# def extract_features(img_path):
#     img = Image.open(img_path).convert('RGB')
#     img_tensor = preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         x = model.stem(img_tensor)
#         for stage in model.stages:
#             x = stage(x)                # Cho qua từng stage
#         x = torch.sum(x, dim=[2, 3])
#         print(x.shape)  # Sau dòng x = torch.sum(...)
#         # Sum-pooling → SPoC
#         x = x / torch.norm(x, dim=1, keepdim=True)  # Normalize
#     return x.cpu().numpy()[0]

def extract_features(img_path):
        # Mở và tiền xử lý ảnh
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # Đảm bảo mô hình ở chế độ eval
        model.eval()
        
        with torch.no_grad():
            # Qua stem
            x = model.stem(img_tensor)
            print(f"After stem: {x.shape}")

            # Lặp qua tất cả stage, chỉ lấy stage cuối
            for i, stage in enumerate(model.stages):
                x = stage(x)
                # print(f"After stage {i+1}: {x.shape}")

            # Lấy stage cuối (đã được xử lý ở bước trên, x là đầu ra của stage cuối)
            # Lặp qua các block trong stage cuối
            for j, block in enumerate(model.stages[-1].blocks):
                x = block(x)
                # print(f"After block {j+1} in final stage: {x.shape}")

            # Sum pooling (SPoC)
            x = torch.sum(x, dim=[2, 3])  # Tổng hợp theo chiều không gian
            x = x / torch.norm(x, dim=1, keepdim=True)  # Chuẩn hóa L2
            # print(f"Final feature shape: {x.shape}")

        return x.cpu().numpy()[0]


# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Extract features and store in Feature table
features_by_class = defaultdict(list)
image_ids_by_class = defaultdict(list)
all_features = []

for class_folder in os.listdir(IMG_FOLDER):
    class_path = os.path.join(IMG_FOLDER, class_folder)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                research = Research.objects.get(image_field_name=f'{img_name}')
                feat = extract_features(img_path)
                features_by_class[class_folder].append(feat)
                image_ids_by_class[class_folder].append(research.image_id)
                all_features.append(feat)
                
                Feature.objects.create(
                    image=research,
                    model_name=MODEL_TYPE,
                    feature_vector=feat.tobytes()
                )
                print(f'✅ {img_name} - class: {class_folder}, image_id: {research.image_id}')
            except Research.DoesNotExist:
                print(f'⚠️ {img_name}: Not found in Research table')
            except Exception as e:
                print(f'⚠️ {img_name}: {e}')

# Train PCA and store PCA-transformed features
all_features = np.array(all_features)
if len(all_features) > 0:
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(all_features)
    joblib.dump(pca, os.path.join(INDEX_DIR, f'pca_{MODEL_TYPE}.pkl'))
    print('🎉 PCA đã được huấn luyện và lưu.')
    
    Feature.objects.filter(model_name=MODEL_TYPE).delete()
    for class_name in class_names:
        if class_name in features_by_class:
            for feat, image_id in zip(features_by_class[class_name], image_ids_by_class[class_name]):
                try:
                    research = Research.objects.get(image_id=image_id)
                    pca_feat = pca.transform(feat.reshape(1, -1))[0]
                    Feature.objects.create(
                        image=research,
                        model_name=MODEL_TYPE,
                        feature_vector=pca_feat.tobytes()
                    )
                    print(f'✅ Stored PCA feature for image_id: {image_id}')
                except Exception as e:
                    print(f'⚠️ Error storing PCA feature for image_id {image_id}: {e}')
    
    # Create and save FAISS indices
    # for class_name in class_names:
    #     if class_name in features_by_class:
    #         vectors = np.array([pca.transform(feat.reshape(1, -1))[0] for feat in features_by_class[class_name]]).astype('float32')
    #         index = faiss.IndexFlatL2(PCA_COMPONENTS)
    #         index.add(vectors)
            
    #         faiss.write_index(index, os.path.join(INDEX_DIR, f'{MODEL_TYPE}_class_{class_name}.index'))
    #         joblib.dump(image_ids_by_class[class_name],
    #                    os.path.join(INDEX_DIR, f'{MODEL_TYPE}_image_ids_{class_name}.pkl'))
    #         print(f'✅ Đã tạo và lưu FAISS index cho lớp {class_name}.')
    for class_name in class_names:
        if class_name in features_by_class:
            pca_vectors = []
            for feat in features_by_class[class_name]:
                pca_feat = pca.transform(feat.reshape(1, -1))[0]
                pca_vectors.append(pca_feat)

            vectors = np.array(pca_vectors).astype('float32')
            faiss.normalize_L2(vectors)  # chuẩn hóa để dùng cosine similarity
            index = faiss.IndexFlatIP(vectors.shape[1])  # dùng inner product = cosine similarity sau normalize
            index.add(vectors)

            # Save FAISS index và danh sách image_ids
            faiss.write_index(index, os.path.join(INDEX_DIR, f'{MODEL_TYPE}_class_{class_name}.index'))
            joblib.dump(image_ids_by_class[class_name],
                        os.path.join(INDEX_DIR, f'{MODEL_TYPE}_image_ids_{class_name}.pkl'))
            print(f'✅ Đã tạo và lưu FAISS index cho lớp {class_name}.')

else:
    print('⚠️ Không có đặc trưng nào được trích xuất.')