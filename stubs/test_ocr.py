import cv2
import sys
from stubs.ocr_recognizer import recognize_plate_text

if len(sys.argv) < 2:
    print("Использование: python test_ocr.py <путь_к_изображению>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
    print(f"Не удалось загрузить изображение: {img_path}")
    sys.exit(1)

print(f"Изображение загружено: {img.shape}")
result = recognize_plate_text(img)
print(f"\n{'='*50}")
print(f"Результат распознавания: '{result}'")
print(f"{'='*50}")