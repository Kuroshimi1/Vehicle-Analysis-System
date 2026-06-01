from ultralytics import YOLO
import numpy as np
from typing import List, Tuple
import os

# Классы транспортных средств в датасете COCO
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = 0.395

# Получаем путь к папке, где лежит этот файл (stubs)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Формируем путь к модели, которая лежит в той же папке
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

CAR_MODEL = YOLO(MODEL_PATH)


def detect_cars(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    results = CAR_MODEL(
        frame,
        device='cpu',
        verbose=False,
        classes=list(VEHICLE_CLASSES),
        conf=CONFIDENCE_THRESHOLD        # ← минимальная уверенность
    )

    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            bboxes.append((x1, y1, x2, y2))

    return bboxes