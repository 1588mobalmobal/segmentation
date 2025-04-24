import torch
from transformers import SegformerForSemanticSegmentation
import os
import time
import cv2
import numpy as np
import torch.nn.functional as F
import albumentations as A

dir = os.path.dirname(__file__)

# 클래스 매핑 (학습 시 사용한 것과 동일해야 함)
class_mapping = {
    'ignore': 0,
    'land-green': 1,
    'land-rock': 2,
    'land-dry': 3,
    'land-flat': 4,
    'water': 5,
    'sky' : 6,
    'end_of_world': 7,
    'tree': 8,
    'cannon': 9,
    'tank': 10,
    'mountain': 11,
    'house': 12,
    'rock': 13,
    'shadow': 14
}

model = None

def init_model():
    global model
    if model == None:
        # 모델 초기화
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            num_labels=len(class_mapping),
            ignore_mismatched_sizes=True
        ).to("cuda")
        # 학습된 가중치 로드
        model.load_state_dict(torch.load(os.path.join(dir, "segformer_model.pth")))
        model.eval()  # 추론 모드 활성화
        return model
    
# 전처리 파이프라인
transform = A.Compose([
    A.Resize(512, 1024),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    image_transformed = transformed["image"]
    image_tensor = torch.tensor(image_transformed).permute(2, 0, 1).float().unsqueeze(0).to("cuda")
    return image_tensor, image_rgb

def predict_segmentation(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor).logits
        outputs = F.interpolate(outputs, size=(512, 1024), mode="bilinear", align_corners=False)
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    return preds