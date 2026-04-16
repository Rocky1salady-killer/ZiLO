import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import cv2
import torch
from ultralytics import YOLO



def main():
    
    model_path = "weights\ZiLOBase-coco.pt"  
    model = YOLO(model_path)

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Using device: {device}")

    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        
        results = model.predict(source=frame, device=device, imgsz=640, conf=0.5)

        
        for result in results:
            frame = result.plot(font_size=10, line_width=2)

        
        cv2.imshow("LeYOLO Real-time Detection", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
