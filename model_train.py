from ultralytics import YOLO
import torch
import os


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

    model = YOLO("yolo11n-pose.pt")

    if torch.cuda.is_available():
        model.train(
            data="hand-keypoints.yaml", 
            epochs=25, 
            imgsz=640, 
            device=device,
            hsv_h=0.03,
            hsv_s=0.6,
            hsv_v=0.5,
            translate=0.5,
            scale=0.5,
            fliplr=0.5,
            erasing=0.3,
            degrees=180
            )

    files = os.listdir("./Models")
    file_idx = [int((f[17:])[:-3]) for f in files]
    model.save(f"./Models/gesture_tracker_v{(file_idx.sort())[-1] + 1}.pt")