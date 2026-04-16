import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 

from ultralytics import YOLO

def train_model():
    model = YOLO("runs/ablation4/zilo5tiny_tag_123/weights/best.pt")

    
    model.val()

