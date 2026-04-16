import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

model = YOLO("weights\ZiLOBase-coco.pt")  
model.train
model.export(format="engine", half=True)  
