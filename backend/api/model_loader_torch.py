import torch
from torchvision import models
import timm

import torch.nn as nn
import torch.optim as optim

import timm
import torch.nn as nn
import torch
from torch.optim import AdamW

def build_convnextv2_raw_model(num_classes, lr=0.000729,
                l2_reg=0.000025, fine_tune_all=False, unfreeze_from_stage=3):

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
        nn.AdaptiveAvgPool2d(1),   # (B, C, H, W) → (B, C, 1, 1)
        nn.Flatten(1),             # (B, C, 1, 1) → (B, C)
        nn.Linear(in_features, 11) # Linear classifier
    )

    return model



def build_convnext_v2_aug_model(num_classes, lr=0.000466, 
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



def load_pytorch_model(model_name, weight_path, num_classes):
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    else:
        raise ValueError("Only alexnet and convnextv2 supported here.")

    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model


def load_convnextv2_model(model_path, num_classes, model_type):
    if model_type == "convnext_v2_raw":
        model = build_convnextv2_raw_model(num_classes=num_classes)
    else:
        model = build_convnext_v2_aug_model(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    # print(checkpoint['model_state_dict'].keys())

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    return model




# def load_alexnet_model( path, num_classes, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load AlexNet với trọng số ImageNet ban đầu
#     weights = models.AlexNet_Weights.IMAGENET1K_V1
#     model = models.alexnet(weights=weights)

#     # Tùy biến lại phần classifier để phù hợp với num_classes đã train
#     model.classifier = nn.Sequential(
#         nn.Linear(256 * 6 * 6, 256),
#         nn.ReLU(),
#         nn.Dropout(p=0.189303),
#         nn.Linear(256, num_classes)
#     )

#     # Load trọng số đã huấn luyện
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.to(device)
#     model.eval()  

#     return model
def load_alexnet_model(path, num_classes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = models.AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights)

    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 256),  # Sử dụng 256 units cho lớp đầu tiên của classifier
        nn.ReLU(),
        nn.Dropout(p=  0.204257  ),  # Dropout rate 0.189303
        nn.Linear(256, num_classes)  # Lớp cuối cùng để phân loại với số lượng lớp output là num_classes
    )

    # Đóng băng layers trước layer thứ 4
    for param in model.features.parameters():
        param.requires_grad = False

    # Mở khóa các layer từ layer thứ 4
    for i in range(2, len(model.features)):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Đảm bảo rằng các parameters của classifier luôn được huấn luyện
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Tạo optimizer với một weight_decay cho tất cả các tham số
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.000114, 
        weight_decay=0.000402                   
    )


    # Load checkpoint và lấy đúng phần model_state_dict 
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Trong trường hợp chỉ chứa state_dict

    model.to(device)
    model.eval()

    return model
  