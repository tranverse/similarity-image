import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model, Model
import django, os, sys

# Setup Django
sys.path.append("E:/similarity_image")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.similarity_image.settings")
django.setup()
from backend.api.models import Research, Feature

# Load mô hình và PCA
vgg16_aug = load_model('E:/similarity_image/models/vgg16/vgg16_aug2_best_params.keras')
spoc_extractor = Model(inputs=vgg16_aug.input,
                       outputs=vgg16_aug.get_layer("block5_pool").output)
pca = joblib.load("pca_256.pkl")

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = spoc_extractor.predict(img_array)
    pooled = np.sum(features, axis=(1, 2))
    normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    return normalized

# Lưu vào CSDL
img_folder = r"E:\LuanVan\data\raw"
model_name = "vgg16_aug"

pca = joblib.load("pca_256.pkl")

for class_folder in os.listdir(img_folder):
    class_path = os.path.join(img_folder, class_folder)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                feat = extract_features(img_path)
                reduced = pca.transform(feat.reshape(1, -1))
                research = Research.objects.filter(image_field_name=img_name).first()
                if research:
                    Feature.objects.create(
                        image=research,
                        model_name="vgg16_aug",
                        feature_vector=reduced.tobytes()
                    )
                    print(f"✅ Đã lưu đặc trưng cho {img_name}")
                else:
                    print(f"❌ Không tìm thấy bản ghi cho {img_name}")
            except Exception as e:
                print(f"⚠️ Lỗi {img_name}: {e}")
