import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

# CLAHE и морфологическое ядро — создаём один раз
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

PLATE_DETECTOR_MODEL = None

# Ищем модель относительно этого файла
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "plate_detector.pt")


def load_plate_model():
    global PLATE_DETECTOR_MODEL
    if not os.path.exists(_MODEL_PATH):
        logger.warning(f"Модель не найдена: {_MODEL_PATH}. Будет использован OpenCV.")
        return
    try:
        from ultralytics import YOLO
        PLATE_DETECTOR_MODEL = YOLO(_MODEL_PATH)
        PLATE_DETECTOR_MODEL(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        logger.info("✅ Модель детектора номеров загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        PLATE_DETECTOR_MODEL = None


def _detect_plates_opencv(car_roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Улучшенный OpenCV-детектор номерных знаков."""
    h, w = car_roi.shape[:2]

    # Работаем с нижними 60% изображения — номер обычно там
    search_top = int(h * 0.4)
    roi = car_roi[search_top:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()

    # Улучшение контраста
    gray = _CLAHE.apply(gray)

    # Адаптивная бинаризация работает лучше equalizeHist для номеров
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = roi.shape[0] * roi.shape[1]
    best_plate, best_score = None, -1

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if ch == 0 or cw < 60 or ch < 15:
            continue

        aspect = cw / ch
        area_ratio = (cw * ch) / img_area

        # Критерии типичного номерного знака
        score = 0
        if 2.0 <= aspect <= 6.0:
            score += 2
        if 0.01 <= area_ratio <= 0.25:
            score += 2
        if cw >= 80 and ch >= 20:
            score += 1
        # Проверяем насыщенность цвета (номера обычно светлые)
        plate_crop = gray[y:y + ch, x:x + cw]
        if plate_crop.size > 0 and np.mean(plate_crop) > 100:
            score += 1

        if score > best_score:
            best_score = score
            best_plate = (x, y + search_top, x + cw, y + search_top + ch)

    if best_plate and best_score >= 4:
        return best_plate

    # Дефолт: нижняя центральная часть авто
    return (int(w * 0.25), int(h * 0.65), int(w * 0.75), int(h * 0.85))


def detect_plates(car_roi: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if car_roi is None or car_roi.size == 0:
        return None

    if PLATE_DETECTOR_MODEL is not None:
        try:
            results = PLATE_DETECTOR_MODEL(
                car_roi, device='cpu', verbose=False, conf=0.45
            )
            best_plate, best_conf = None, 0.0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    if (x2 - x1) > 50 and (y2 - y1) > 15 and conf > best_conf:
                        best_conf = conf
                        best_plate = (x1, y1, x2, y2)

            if best_plate:
                print(f"[Plate Detector] YOLO нашел номер с уверенностью {best_conf:.2f}: {best_plate}")
                return best_plate
            else:
                print(f"[Plate Detector] YOLO не нашел номер, используем OpenCV метод")
        except Exception as e:
            logger.error(f"Ошибка YOLO детекции: {e}")

    result = _detect_plates_opencv(car_roi)
    if result:
        print(f"[Plate Detector] OpenCV нашел номер: {result}")
    else:
        print(f"[Plate Detector] OpenCV не нашел номер")
    return result


def enhance_plate_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    plate_roi = image[y1:y2, x1:x2]
    if plate_roi.size == 0:
        return None

    scale = max(2.0, 400.0 / plate_roi.shape[1])
    new_w = int(plate_roi.shape[1] * scale)
    new_h = int(plate_roi.shape[0] * scale)
    plate_roi = cv2.resize(plate_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if plate_roi.ndim == 3:
        lab = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = _CLAHE.apply(l)
        plate_roi = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return plate_roi


load_plate_model()