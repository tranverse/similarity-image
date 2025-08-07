import numpy as np
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from PIL import Image
import faiss
import joblib
from sklearn.decomposition import PCA
from django.conf import settings
from .models import Research
import os
from PIL import Image
from tensorflow.keras.preprocessing import image

# Model paths
MODEL_PATHS = {
    'vgg16': 'E:/similarity_image/models/vgg16/vgg16_raw.keras',
    'vgg16_aug': r'E:\similarity_image\models\vgg16\vgg16_aug_best_params_final.keras',
    'convnext_v2_aug': r"E:\convnext_v2_best_params_aug_final_11_7.pth",
    'alexnet_aug': r"E:\alexnet_best_params_aug_final.pth"
}

# Directory for FAISS indices and PCA
INDEX_DIR = r'E:\similarity_image\extract_features'
IMG_FOLDER = r'E:\similarity_image\dataset'

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

# PCA configuration
PCA_COMPONENTS = 256
pca = None

# FAISS indices for each class and model
dimension = PCA_COMPONENTS
faiss_indices = {model_type: {class_name: None for class_name in class_names} for model_type in MODEL_PATHS.keys()}

def get_preprocess_torch(resize_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def preprocess_keras_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array

def extract_spoc_features(img, model, model_type):
    """Extract SPoC features from an image."""
    if model_type.startswith('vgg16'):
        spoc_extractor = Model(inputs=model.input, outputs=model.get_layer('block5_conv3').output)

        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  

        features = spoc_extractor.predict(img_array)
        print(f"Raw shape from block5_pool (VGG16):", features.shape)  # (1, 14, 14, 512)
        pooled = np.sum(features, axis=(1, 2))
        normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalized[0]
    elif model_type.startswith('convnext_v2'):
        preprocess = get_preprocess_torch()  
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # with torch.no_grad():
        #     x = model.stem(img_tensor)
        #     for stage in model.stages:
        #         x = stage(x)  

        #     print(f"Raw shape from ConvNeXt stage output:", x.shape)  # (1, 768, 7, 7)
        #     x = torch.sum(x, dim=[2, 3])  # Sum-pooling (SPoC)
        #     x = x / torch.norm(x, dim=1, keepdim=True) 
        # return x.cpu().numpy()[0]
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
    elif model_type.startswith('alexnet'):
        preprocess = get_preprocess_torch(resize_size=(227, 227))  # define elsewhere
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            x = model.features(img_tensor)  # e.g. (1, 256, 6, 6)
            print(f"[AlexNet] Raw shape:", x.shape)
            x = torch.sum(x, dim=[2, 3])    # SPoC
            x = x / torch.norm(x, dim=1, keepdim=True)
        return x.cpu().numpy()[0]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def classify_image(img, model, model_type):
    """Classify an image."""
    if model_type.startswith('vgg16'):
        img_array = preprocess_keras_image(img)
        preds = model.predict(img_array)
        class_id = np.argmax(preds[0])
        confidence = np.max(preds[0])
        return class_names[class_id], confidence, preds[0]
    elif model_type.startswith(('convnext_v2', 'alexnet')):
        preprocess = get_preprocess_torch() 
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            preds = torch.softmax(preds, dim=1)
        class_id = torch.argmax(preds[0]).item()
        confidence = preds[0][class_id].item()
        return class_names[class_id], confidence, preds[0].cpu().numpy()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_pca(model_type):
    """Load PCA model."""
    global pca
    pca_path = os.path.join(INDEX_DIR, f'{model_type}/pca_{model_type}.pkl')
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        raise ValueError(f"PCA model not found for {model_type}")

def apply_pca(features):
    """Apply PCA transformation."""
    if pca is None:
        raise ValueError("PCA model not loaded")
    return pca.transform(features.reshape(1, -1))[0]

def load_faiss_index(model_type, class_name):
    """Load FAISS index for a class."""
    index_path = os.path.join(INDEX_DIR, f'{model_type}/{model_type}_class_{class_name}.index')
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return faiss.IndexFlatL2(dimension)

def save_faiss_index(index, model_type, class_name):
    """Save FAISS index."""
    faiss.write_index(index, os.path.join(INDEX_DIR, f'{model_type}/{model_type}_class_{class_name}.index'))

def load_image_ids(model_type, class_name):
    """Load image IDs for a class."""
    ids_path = os.path.join(INDEX_DIR, f'{model_type}/{model_type}_image_ids_{class_name}.pkl')
    if os.path.exists(ids_path):
        return joblib.load(ids_path)
    return []

def save_image_ids(image_ids, model_type, class_name):
    """Save image IDs."""
    joblib.dump(image_ids, os.path.join(INDEX_DIR, f'{model_type}/{model_type}_image_ids_{class_name}.pkl'))

# def search_similar_images(img, model, model_type, threshold, top_k=100):
#     """Search for similar images in the predicted class."""
#     pred_class, confidence, preds = classify_image(img, model, model_type)
#     load_pca(model_type)
#     features = extract_spoc_features(img, model, model_type)
#     query_vector = apply_pca(features)
#     index = load_faiss_index(model_type, pred_class)
#     image_ids = load_image_ids(model_type, pred_class)
#     D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), top_k)
#     similar_items = []
#     for dist, idx in zip(D[0], I[0]):
#         if idx < len(image_ids):
#             sim_score = 1 - dist
#             if sim_score >= threshold:
#                 try:
#                     image_id = image_ids[idx]
#                     research = Research.objects.get(image_id=image_id)
#                     similar_items.append((sim_score, research))
#                 except Research.DoesNotExist:
#                     continue
#     similar_items.sort(reverse=True, key=lambda x: x[0])
#     similar_results = [
#         {
#             'similarity': float(f'{sim:.4f}'),
#             'title': research.title,
#             'doi': research.doi,
#             'caption': research.caption,
#             'image_field_name': research.image_field_name,
#             'authors': research.authors,
#             'language': research.language,
#         }
#         for sim, research in similar_items
#     ]   
#     return {
#         'predicted_class': pred_class,
#         'confidence': f'{confidence * 100:.2f}%',
#         'all_classes': [
#             {'label': class_names[i], 'confidence': float(f'{conf * 100:.2f}')}
#             for i, conf in enumerate(preds)
#         ],
#         'similar_images': similar_results
#     }
def search_similar_images(img, model, model_type, threshold, top_k=100):
    """Search for similar images using cosine similarity (no PCA for AlexNet)."""
    pred_class, confidence, preds = classify_image(img, model, model_type)
    features = extract_spoc_features(img, model, model_type)

    if model_type.lower() == 'alexnet_aug':
        query_vector = features / np.linalg.norm(features)
    else:
        load_pca(model_type)
        query_vector = apply_pca(features)
        query_vector = query_vector / np.linalg.norm(query_vector)

    index = load_faiss_index(model_type, pred_class)
    image_ids = load_image_ids(model_type, pred_class)

    # D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), top_k)
    D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), index.ntotal)

    similar_items = []
    for sim_score, idx in zip(D[0], I[0]):
        if idx < len(image_ids) and sim_score >= threshold:
            try:
                image_id = image_ids[idx]
                research = Research.objects.get(image_id=image_id)
                similar_items.append((sim_score, research))
            except Research.DoesNotExist:
                continue

    similar_items.sort(reverse=True, key=lambda x: x[0])
    similar_results = [
        {
            'similarity': float(f'{sim:.4f}'),
            'title': research.title,
            'doi': research.doi,
            'caption': research.caption,
            'image_field_name': research.image_field_name,
            'authors': research.authors,
            'language': research.language,
            'page_number': research.page_number,
            'approved_date': research.approved_date     
        }
        for sim, research in similar_items
    ]   

    return {
        'predicted_class': pred_class,
        'confidence': f'{confidence * 100:.2f}%',
        'all_classes': [
            {'label': class_names[i], 'confidence': float(f'{conf * 100:.2f}')}
            for i, conf in enumerate(preds)
        ],
        'similar_images': similar_results
    }

# Device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
