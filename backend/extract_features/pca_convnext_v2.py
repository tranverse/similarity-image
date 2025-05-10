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
MODEL_PATH = r'E:\similarity_image\models\convnextv2\convnext_v2_best_params_aug2_final.pth'
INDEX_DIR = 'E:/similarity_image/extract_features'
IMG_FOLDER = 'E:/LuanVan/data/raw'
MODEL_TYPE = 'convnext_v2_aug'
PCA_COMPONENTS = 256

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

# Define model building function
def build_model(num_classes, lr=0.000169, dropout_rate=0.230522, dense_units=576,
                l2_reg=1.664343e-07, fine_tune_all=False, unfreeze_from_stage=2):
    model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True, num_classes=0)
    in_features = model.head.in_features
    for param in model.parameters():
        param.requires_grad = False
    for i, stage in enumerate(model.stages):
        if i >= unfreeze_from_stage:
            for param in stage.parameters():
                param.requires_grad = True
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Keep original head for classification
        nn.Flatten(),
        nn.Linear(in_features, dense_units),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, num_classes)
    )
    return model

# Load ConvNeXt V2 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(num_classes=len(class_names))
checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# Define feature extractor (up to last stage, no pooling)
feature_extractor = nn.Sequential(
    model.stem,
    model.stages  # Up to stage 3
)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path):
    """Extract SPoC features from an image using ConvNeXt V2."""
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_map = feature_extractor(img_tensor)  # [batch, C, H, W]
        # Sum-pooling over spatial dimensions
        features = torch.sum(feature_map, dim=[2, 3])  # [batch, C]
        normalized = features / torch.norm(features, dim=1, keepdim=True)
    return normalized.cpu().numpy()[0]

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
    for class_name in class_names:
        if class_name in features_by_class:
            vectors = np.array([pca.transform(feat.reshape(1, -1))[0] for feat in features_by_class[class_name]]).astype('float32')
            index = faiss.IndexFlatL2(PCA_COMPONENTS)
            index.add(vectors)
            
            faiss.write_index(index, os.path.join(INDEX_DIR, f'{MODEL_TYPE}_class_{class_name}.index'))
            joblib.dump(image_ids_by_class[class_name],
                       os.path.join(INDEX_DIR, f'{MODEL_TYPE}_image_ids_{class_name}.pkl'))
            print(f'✅ Đã tạo và lưu FAISS index cho lớp {class_name}.')
else:
    print('⚠️ Không có đặc trưng nào được trích xuất.')