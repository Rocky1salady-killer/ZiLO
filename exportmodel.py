import os
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


model_dirs = [
    "yoursmodel",
]


for model_dir in model_dirs:
    pt_path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(pt_path):
        print(f"❌ Skipping {pt_path}: file not found.")
        continue

    print(f"\n🚀 Exporting model: {pt_path}")
    try:
        model = YOLO(pt_path)
        model.export(
            format="engine",
    	    dynamic=False,
            imgsz=640,
            batch=1,
            workspace=4,
            half=True,         # FP16
            device=0
        )
        print(f"✅ Exported: {pt_path}")
    except Exception as e:
        print(f"❌ Failed to export {pt_path}: {e}")
