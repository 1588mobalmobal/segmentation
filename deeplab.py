import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.amp import GradScaler, autocast # 학습 속도 가속화
import torch.nn.functional as F
from torchmetrics import JaccardIndex #
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import numpy as np
import albumentations as A # 데이터 증강을 위함
from PIL import Image
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

class_mapping = {
    0 :'void',
    1 :'dirt',
    2 :'sand',
    3 :'grass' ,
    4 :'tree' ,
    5 :'obstacle' ,
    6 :'water' ,
    7 :'sky' ,
    8 :'vehicle' ,
    9 :'person' ,
    10 :'hard_surface',
    11 : 'gravel',
    12 :'vegetation' ,
    13 :'mulch',
    14 :'rock'
}

# 분류된 클래스별로 색상을 할당합니다 
class_to_rgb_map = {
    0 : (0, 0, 0),
    1 : (108, 64, 20),
    2 : (255, 229, 204),
    3 : (0, 102, 0),
    4 : (0, 255, 0),
    5 : (0, 153, 153),
    6 : (0, 128, 255),
    7 : (0, 0, 255),
    8 : (255, 255, 0),
    9 : (255, 0, 127),
    10 : (64, 64, 64),
    11 : (100, 110, 50),
    12 : (183, 255, 0),
    13 : (153, 76, 0),
    14 : (160, 160, 160),
}

# 소스 이미지를 읽어서 클래스를 분류합니다 
rgb_to_class_map = {
    (0, 0, 0): 0, 
    (108, 64, 20): 1, 
    (255, 229, 204): 2, 
    (0, 102, 0): 3, 
    (0, 255, 0): 4, 
    (0, 153, 153): 5, 
    (0, 128, 255): 6, 
    (0, 0, 255): 7, 
    (255, 255, 0): 8, 
    (255, 0, 127): 5, 
    (64, 64, 64): 10, 
    (255, 128, 0): 11, 
    (255, 0, 0): 5, 
    (153, 76, 0): 13, 
    (102, 102, 0): 11, 
    (102, 0, 0): 12, 
    (0, 255, 128): 8, 
    (204, 153, 255): 9, 
    (102, 0, 204): 5, 
    (255, 153, 204): 12, 
    (0, 102, 102): 5, 
    (153, 204, 255): 14, 
    (102, 255, 255): 5, 
    (101, 101, 11): 5, 
    (114, 85, 47): 5,
    (170, 170, 170): 5,
    (41, 121, 255): 5,
    (101, 31, 255): 10,
    (137, 149, 9): 10,
    (134, 255, 239): 1,
    (99, 66, 34): 1,
    (110, 22, 138): 5,
    }

directory = os.getcwd()
device = "cuda" if torch.cuda.is_available() else "cpu"

mean= [-0.02662486, -0.01916305, -0.00590634]
std= [0.07481168, 0.07667251, 0.07697445]

def init_model():
    preprocessor = AutoImageProcessor.from_pretrained(
        "google/deeplabv3_mobilenet_v2_1.0_513",
        size={'height': 512, 'width': 512},
        image_mean=mean, image_std=std, do_reduce_labels=False
        )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "google/deeplabv3_mobilenet_v2_1.0_513",
        num_labels=len(class_mapping),  # 매핑 수만큼 
        ignore_mismatched_sizes=True,    # 클래스 수 불일치 무시
        )
    state_dict = torch.load(os.path.join(directory, "deeplabv3_epoch21.pth"))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    model.config.image_size = 512
    # model.classifier = torch.nn.Conv2d(320, 15, kernel_size=1)
    model.to(device)
    print('😀 DeepLabV3+ has been Initialized!!💚')
    # model = torch.compile(model)
    # print(f"Number of labels in model config: {model.config.num_labels}")
    # print(preprocessor.__dict__)
    return model, preprocessor

def rgb_to_class_index_optimized(mask_rgb, color_map):
    mask_rgb = mask_rgb.astype(np.uint8)
    H, W, _ = mask_rgb.shape
    mask_idx = np.zeros((H, W), dtype=np.uint8)
    color_map_array = np.array(list(color_map.keys()), dtype=np.uint8)  # [N, 3]
    indices = np.array(list(color_map.values()), dtype=np.uint8)  # [N]
    for i in range(len(color_map_array)):
        mask = np.all(mask_rgb == color_map_array[i], axis=-1)
        mask_idx[mask] = indices[i]
    return mask_idx

def predict_segmentation(image_path, model, preprocessor):
    sample_image = np.array(Image.open(image_path))
    inputs = preprocessor(images=sample_image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        # print(f"Probs shape: {probs.shape}")  # [batch_size, 25, 512, 512]
        # print(f"Probs max: {probs.max().item()}, min: {probs.min().item()}")
        for class_idx in range(15):
            class_prob = probs[0, class_idx, 0, 0]  # 첫 번째 픽셀의 클래스별 확률
            # print(f"Class {class_idx} probability: {class_prob.item()}")
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # [1, 512, 512]
        prediction = predictions[0]
    
    return prediction

def visualize_segmentation(image_path, prediction, output_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512))  # 모델 입력 크기에 맞춤

    # 클래스 인덱스를 RGB로 변환
    seg_map = np.zeros((512, 512, 3), dtype=np.uint8)
    color_array = np.array([class_to_rgb_map.get(i, (0, 0, 0)) for i in range(25)], dtype=np.uint8)
    seg_map = color_array[prediction]

    unique_classes = np.unique(prediction)
    legend_elements = [
            Patch(facecolor=np.array(class_to_rgb_map[idx])/255, label=class_mapping.get(idx, f"Class {idx}"))
            for idx in unique_classes if idx in class_to_rgb_map
        ]

    # 원본 이미지와 세그멘테이션 맵 시각화

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(seg_map)
    ax2.set_title("Segmentation Map")
    ax2.axis("off")

    # 범례 추가
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close(fig)