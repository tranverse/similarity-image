import fitz  # PyMuPDF
import base64
from io import BytesIO
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.http import HttpResponse
from tensorflow.keras.models import load_model
import torch
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from rest_framework.response import Response
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from .model_loader_torch import *
import io
import os
model_dir = 'E:/similarity_image/models'
train_dir_raw = 'E:/LuanVan/data/split-raw/train'
val_dir_raw = 'E:/LuanVan/data/split-raw/val'
test_dir_raw = 'E:/LuanVan/data/split-raw/test'
test_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  
)
test_generator_raw = test_datagen.flow_from_directory( 
    test_dir_raw,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Không trộn dữ liệu khi kiểm tra (test)
)
class_names = test_generator_raw.class_indices.keys()
class_names = list(test_generator_raw.class_indices.keys())


vgg16_raw = load_model(r'E:\similarity_image\models\vgg16\vgg16_raw_best_params_final.keras', compile=False)
vgg16_aug = load_model(r'E:\similarity_image\models\vgg16\vgg16_best_params_aug_final.keras', compile=False)

convnext_v2_raw_path = r'E:\similarity_image\models\convnextv2\convnext_v2_best_params_raw_final.pth'
convnext_v2_aug_path = r'E:\similarity_image\models\convnextv2\convnext_v2_best_params_aug_final.pth'

alexnet_raw_path = r'E:\similarity_image\models\alexnet\Alexnet_best_params_raw.pth'
alexnet_aug_path = r'E:\similarity_image\models\alexnet\Alexnet_best_params_raw.pth'
# Preprocessing for PyTorch Models
def preprocess_pytorch_image(img, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# Preprocessing for Keras Models (VGG16)
def preprocess_keras_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main(request):
    return HttpResponse("Hello")

class ExtractImagesFromPdfView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        pdf_file = request.FILES.get('pdf')
        if not pdf_file:
            return JsonResponse({'error': 'No PDF file provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            images_data = []
            original_name = os.path.splitext(pdf_file.name)[0]
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

            global_img_index = 1
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pdf_page_number = page.number + 1
                image_list = page.get_images(full=True)

                # Dùng để tìm page number dưới cùng
                page_dict = page.get_text("dict")
                blocks = page_dict["blocks"]
                page_number_found = None

                for block in blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text.isdigit():
                                    x0, y0, x1, y1 = span["bbox"]
                                    if y0 > 750 and (x0 > 400 or (250 < x0 < 350)):
                                        page_number_found = text

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    width = img[1]
                    height = img[2]
                    width = base_image["width"]
                    height = base_image["height"]

                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        img_np = np.array(pil_img)
                        
                        # Kiểm tra kích thước ảnh
                        too_small = width < 50 or height < 50
                        
                        # Kiểm tra monochrome (chỉ có một màu)
                        std_color = np.std(img_np)
                        is_monochrome = std_color < 10  # Tăng giá trị này nếu cần xác định ảnh đơn sắc rõ ràng hơn
                        
                        # Kiểm tra ảnh hợp lệ: Kích thước và không phải monochrome
                        is_valid = not (too_small or is_monochrome)
                        
                        # Debug thông báo
                        if not is_valid:
                            print(f"Image invalid: width={width}, height={height}, std_color={std_color}, monochrome={is_monochrome}, too_small={too_small}")
                        
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        is_valid = False


                    # Tính trung tâm ảnh để tìm caption
                    image_center_x = width / 2

                   

                    # Duyệt qua tất cả các block để lấy tọa độ của ảnh và văn bản
                    for block in blocks:
                        if block["type"] == 1:  # Nếu block là ảnh (type = 1)
                            # Lấy tọa độ của ảnh từ bbox
                            image_bbox = block["bbox"]
                            ix0, iy0, ix1, iy1 = image_bbox
                            # print(f"Image bbox: {image_bbox}")  # In tọa độ ảnh ra để kiểm tra

                            # Tìm caption dưới ảnh (dòng văn bản)
                            caption = ""
                            min_distance = float('inf')

                            for tblock in blocks:
                                if tblock["type"] == 0:  # Nếu block là văn bản (type = 0)
                                    for line in tblock["lines"]:
                                        line_text = ""
                                        line_y = None
                                        for span in line["spans"]:
                                            sx0, sy0, sx1, sy1 = span["bbox"]
                                            if line_y is None:
                                                line_y = sy0  # Lấy tọa độ y của dòng văn bản

                                            line_text += span["text"].strip() + " "

                                        # Kiểm tra xem dòng văn bản có nằm dưới ảnh không
                                        if line_y and line_y > iy1:  # Dòng văn bản phải nằm dưới ảnh (y > iy1)
                                            distance = line_y - iy1  # Khoảng cách giữa ảnh và văn bản
                                            if distance <= 50:  # Tìm caption trong phạm vi 50px dưới ảnh
                                                caption = line_text.strip()
                                                break
                                    if caption:
                                        break

                            # Kiểm tra kết quả
                            if caption:
                                print(f"Found caption: {caption}")
                            else:
                                print("No caption found.")



                    b64_image = base64.b64encode(image_bytes).decode('utf-8')
                    image_filename = f"{original_name}_page_{pdf_page_number}_img_{global_img_index}.png"

                    images_data.append({
                        'index': global_img_index,
                        'name': image_filename,
                        'base64': b64_image,
                        'caption': caption,
                        'page_number_position': page_number_found,
                        'is_valid': is_valid
                    })

                    global_img_index += 1

            return JsonResponse({'images': images_data}, status=status.HTTP_200_OK)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

      
# convnextv2_raw = 

print(class_names)
class ClassifyImagesView(APIView):
    def post(self, request):
        image_list = request.data.get("images", [])
        model_type = request.data.get("model")  

        # Load model
        try:
            if model_type == "vgg16":   
                model = vgg16_raw
            elif model_type == "vgg16_aug":
                model = vgg16_aug
            elif model_type == "convnext_v2":
                model = load_convnextv2_model(convnext_v2_raw_path, 11, model_type="convnext_v2")
                model.eval()
            elif model_type == "convnext_v2_aug":
                model = load_convnextv2_model(convnext_v2_aug_path, 11, model_type="convnext_v2_aug")
            elif model_type == "alexnet":
                model = load_alexnet_model(alexnet_raw_path, 11)
            elif model_type == "alexnet_aug":
                model = load_alexnet_model(alexnet_aug_path, 11)

        except Exception as e:
            return Response({"error": f"Model load failed: {e}"}, status=500)

        # Tạo map class id -> class name
        inv_label_map = {v: k for k, v in test_generator_raw.class_indices.items()}
        if isinstance(image_list, str):
                try:
                    image_list = json.loads(image_list) 
                    print(f"Successfully parsed image_list as JSON, number of images: {len(image_list)}")
                except json.JSONDecodeError as e:
                    return Response({"error": f"Error decoding JSON: {e}"}, status=400)
        

        results = []
        for image_obj in image_list:
            try:
                b64_image = image_obj.get("base64")
                print(b64_image)
                img_bytes = base64.b64decode(b64_image)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")

                if model_type == "vgg16" or model_type == "vgg16_aug":
                    img_array = preprocess_keras_image(img)
                    preds = model.predict(img_array)
                    print(preds)
                    class_id = np.argmax(preds)
                    pred_class = class_names[class_id]
                    confidence = f"{np.max(preds)*100:.2f}%"
                    model_type = "vgg16"
                    all_classes = [
                        {"label": class_names[i], "confidence": confidence}
                        for i, conf in enumerate(preds[0])
                    ]
                elif model_type == "convnext_v2" or model_type == "convnext_v2_aug":
                    img = preprocess_pytorch_image(img, target_size=(224, 224))
                    model.eval()
                    with torch.no_grad():
                        # Lấy đầu ra từ mô hình
                        output = model(img)
                        # Tính xác suất cho từng lớp bằng softmax
                        probs = torch.nn.functional.softmax(output[0], dim=0).numpy()    
                        # Lấy lớp có xác suất cao nhất
                        class_id = np.argmax(probs)
                        pred_class = class_names[class_id]          
                        # Lấy giá trị xác suất cho lớp dự đoán
                        confidence = f"{probs[class_id]*100:.2f}%"
                        # Dữ liệu cho tất cả các lớp
                        all_classes = [
                            {"label": class_names[i], "confidence": float(f"{conf*100:.2f}")}
                            for i, conf in enumerate(probs)
                        ]
                elif model_type == "alexnet" or model_type == "alexnet_aug":
                    img = preprocess_pytorch_image(img, target_size=(227, 227))
                    with torch.no_grad():

                        output = model(img.unsqueeze(0))  # batch size = 1
                        predicted = torch.argmax(output, dim=1)

                results.append({
                    "image": b64_image,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "all_classes": all_classes,
                    "model_type": model_type
                })

            except Exception as e:
                results.append({"error": str(e)})
            
            print(f"Type of image_list: {type(results)}")

        return Response({"results": results})

