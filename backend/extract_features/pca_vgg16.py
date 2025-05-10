import os
import numpy as np
import faiss
import joblib
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Model
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
MODEL_PATH = 'E:/similarity_image/models/vgg16/vgg16_aug2_best_params.keras'
INDEX_DIR = 'E:/similarity_image/extract_features'
IMG_FOLDER = 'E:/LuanVan/data/raw'
MODEL_TYPE = 'vgg16_aug'
PCA_COMPONENTS = 256

# Get class names from folder names
class_names = [folder for folder in os.listdir(IMG_FOLDER) if os.path.isdir(os.path.join(IMG_FOLDER, folder))]

# Load model
model = load_model(MODEL_PATH)
spoc_extractor = Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)

def preprocess_keras_image(img_path):
    """Preprocess image from path."""
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img_path):
    """Extract SPoC features."""
    img_array = preprocess_keras_image(img_path)
    features = spoc_extractor.predict(img_array)
    pooled = np.sum(features, axis=(1, 2))
    normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return normalized[0]

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
            # Check if image exists in Research
            try:
                research = Research.objects.get(image_field_name=f'{img_name}')
                feat = extract_features(img_path)
                features_by_class[class_folder].append(feat)
                image_ids_by_class[class_folder].append(research.image_id)
                all_features.append(feat)
                
                # Store in Feature (raw features)
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
    
    # Clear raw features and store PCA-transformed features
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
            
            # Save index and image IDs
            faiss.write_index(index, os.path.join(INDEX_DIR, f'{MODEL_TYPE}_class_{class_name}.index'))
            joblib.dump(image_ids_by_class[class_name],
                       os.path.join(INDEX_DIR, f'{MODEL_TYPE}_image_ids_{class_name}.pkl'))
            print(f'✅ Đã tạo và lưu FAISS index cho lớp {class_name}.')
else:
    print('⚠️ Không có đặc trưng nào được trích xuất.')