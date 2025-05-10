import torch
from torchvision import models
import timm

import torch.nn as nn

import timm
import torch.nn as nn
import torch
from torch.optim import AdamW
def build_convnextv2_raw_model(num_classes, lr=0.000541, dropout_rate=0.112936, dense_units=256,
                l2_reg=0.000081, fine_tune_all=False, unfreeze_from_stage=2):

    model = timm.create_model('convnextv2_tiny.fcmae_ft_in1k', pretrained=True, num_classes=0)
    in_features = model.head.in_features

    # Đóng băng toàn bộ nếu không fine-tune tất cả
    for param in model.parameters():
        param.requires_grad = False

    # Mở các stage từ vị trí chỉ định
    for stage in model.stages[unfreeze_from_stage:]:
        for param in stage.parameters():
            param.requires_grad = True

    # Head mới với các tham số từ trial
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(in_features, dense_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(dense_units, num_classes)
    )

    return model



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
    if model_type == "convnext_v2":
        model = build_convnextv2_raw_model(num_classes=num_classes)
    else:
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

    # Load AlexNet với trọng số ImageNet ban đầu
    weights = models.AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights)

    # Tùy biến lại phần classifier để phù hợp với num_classes đã train
    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 256),
        nn.ReLU(),
        nn.Dropout(p=0.189303),
        nn.Linear(256, num_classes)
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
  