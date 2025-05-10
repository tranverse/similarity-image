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

# Model paths
MODEL_PATHS = {
    'vgg16': 'E:/similarity_image/models/vgg16/vgg16_raw.keras',
    'vgg16_aug': 'E:/similarity_image/models/vgg16/vgg16_aug2_best_params.keras',
    'convnext_v2_aug': 'E:/similarity_image/models/convnext_v2/convnext_v2_best_params_aug2_final',
}

# Directory for FAISS indices and PCA
INDEX_DIR = r'E:\similarity_image\extract_features'
IMG_FOLDER = 'E:/LuanVan/data/raw'

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

# PCA configuration
PCA_COMPONENTS = 256
pca = None

# FAISS indices for each class and model
dimension = PCA_COMPONENTS
faiss_indices = {model_type: {class_name: None for class_name in class_names} for model_type in MODEL_PATHS.keys()}

# Image preprocessing for ConvNeXt V2
preprocess_torch = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_keras_image(img):
    """Preprocess image for Keras model."""
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_spoc_features(img, model, model_type):
    """Extract SPoC features from an image."""
    if model_type.startswith('vgg16'):
        spoc_extractor = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
        img_array = preprocess_keras_image(img)
        features = spoc_extractor.predict(img_array)
        pooled = np.sum(features, axis=(1, 2))
        normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return normalized[0]
    elif model_type.startswith('convnext_v2'):
        feature_extractor = nn.Sequential(
            model.stem,
            model.stages  # Up to stage 3
        )
        img_tensor = preprocess_torch(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_map = feature_extractor(img_tensor)  # [batch, C, H, W]
            features = torch.sum(feature_map, dim=[2, 3])  # Sum-pooling
            normalized = features / torch.norm(features, dim=1, keepdim=True)
        return normalized.cpu().numpy()[0]
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
    elif model_type.startswith('convnext_v2'):
        img_tensor = preprocess_torch(img).unsqueeze(0).to(device)
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

def search_similar_images(img, model, model_type, threshold, top_k=50):
    """Search for similar images in the predicted class."""
    pred_class, confidence, preds = classify_image(img, model, model_type)
    load_pca(model_type)
    features = extract_spoc_features(img, model, model_type)
    query_vector = apply_pca(features)
    index = load_faiss_index(model_type, pred_class)
    image_ids = load_image_ids(model_type, pred_class)
    D, I = index.search(query_vector.astype(np.float32).reshape(1, -1), top_k)
    similar_items = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(image_ids):
            sim_score = 1 - dist
            if sim_score >= threshold:
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