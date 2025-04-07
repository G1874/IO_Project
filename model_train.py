from ultralytics import YOLO
import torch
import os


if __name__ == "__main__":

    model = YOLO("yolo11n-pose.pt")

    if torch.cuda.is_available():
        model.train(
            data="hand-keypoints.yaml",
            epochs=5,
            imgsz=640,
            device='cuda'
        )

    files = os.listdir("./Models")
    file_idx = [int((f[17:])[:-3]) for f in files]
    model.save(f"./Models/gesture_tracker_v{(file_idx.sort())[-1] + 1}.pt")