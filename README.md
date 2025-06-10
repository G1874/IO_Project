# IO_Project - Śledzenie gestów

Projekt umożliwia detekcję i śledzenie punktów charakterystycznych dłoni (keypoints) na obrazach oraz wideo przy użyciu modelu YOLOv8 Pose.

### Trening modelu
Model YOLOv8 jest trenowany z pliku konfiguracyjnego hand-keypoints.yaml:

```bash
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model.train(
    data="hand-keypoints.yaml", 
    epochs=20, 
    imgsz=640, 
    device='cuda',
    erasing=0.4
)

model.save("./Models/gesture_tracker_v2.pt")
```

### Inferencja:
Skrypt inference.py obsługuje zarówno obraz, jak i wideo (również z kamery na żywo).

Obraz:
```bash
python inference.py -i  
```

Kamera lub wideo:
```bash
python inference.py       #wideo z pliku
python inference.py -r    #obraz z kamery
```

Budowanie dockera:
```bash
docker build -t yolo-hand-train .
```

Uruchamianie dockera i treningu sieci:
```bash
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  yolo-hand-train \
  python model_train.py
```

### Resources:
- [YOLO hand-keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)

