import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/cfg/zilobase.yaml")

    model.train(
        data="ultralytics/cfg/datasets/crowdhuman.yaml",
        epochs=5,
        patience=50,
        device="0",
        batch=16,
        imgsz=640,
        resume=False,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        nbs=64,
        workers=16,
        seed=0,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=8.0,
        cls=0.25,
        dfl=1.7,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
        close_mosaic=30,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.2,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        erasing=0.2,
        crop_fraction=1.0,
        multi_scale=False,
        cache=False,
        save_period=50,
    )

    model.val(data="ultralytics/cfg/datasets/crowdhuman.yaml", imgsz=640, max_det=300)
