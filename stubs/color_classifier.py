# stubs/color_classifier.py
import numpy as np
import cv2
from tensorflow import keras
import os

MODEL_PATH = 'trained_color_classifier.h5'
try:
    COLOR_CLASSIFIER = keras.models.load_model(MODEL_PATH)
    print("Color Classifier model loaded successfully.")
except Exception as e:
    print(f"ERROR loading color classifier model ({MODEL_PATH}): {e}. Running in MOCK mode.")
    COLOR_CLASSIFIER = None

# Путь к папке stubs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Путь к модели (если она лежит в корне, идем на уровень выше через os.path.dirname)
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'trained_color_classifier.h5')

IMG_SIZE = (224, 224)
# !!! ВНИМАНИЕ! Убедитесь, что этот список соответствует порядку классов в вашей модели!!!
CLASS_NAMES = ["Black", "Blue", "Brown", "Red", "Gray", "Orange", "Pink", "Red",  "Purple", "White", "Yellow"]

def classify_color(car_roi: np.ndarray) -> tuple[str, float]:
    """Использует TensorFlow модель для классификации цвета."""
    if COLOR_CLASSIFIER is None:
        # --- ПОКА ЗАГЛУШКА ДЛЯ ТЕСТА АРХИТЕКТУРЫ ---
        possible_colors = CLASS_NAMES
        chosen_color = np.random.choice(possible_colors)
        confidence = np.random.uniform(0.6, 0.99)
        return chosen_color, confidence

    # РЕАЛЬНЫЙ КОД (если модель загружена успешно)
    processed_roi = cv2.resize(car_roi, IMG_SIZE)
    input_data = np.expand_dims(processed_roi, axis=0) / 255.0 # Масштабирование 0-1

    # Предсказание (работает на CPU по умолчанию)
    predictions = COLOR_CLASSIFIER.predict(input_data)[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]
    color_label = CLASS_NAMES[predicted_index]
    return color_label, float(confidence)
