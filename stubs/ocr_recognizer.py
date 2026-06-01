from typing import Optional
import numpy as np
import cv2
import easyocr
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ГЛОБАЛЬНЫЕ КЭШИ (создаются один раз) ---
_READER = None

# Кэшируем всё что можно создать заранее
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_KERNEL_SHARPEN = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
_KERNEL_MORPH = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

# Скомпилированные regex
_PATTERN_PLATE = re.compile(r'([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})')
_PATTERN_VALID = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}$')
_PATTERN_VALID_REGION = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

# Единый словарь коррекций
_CORRECTIONS = {
    'B': 'В', 'O': 'О', '0': '0', 'P': 'Р', 'A': 'А',
    'H': 'Н', 'K': 'К', 'M': 'М', 'T': 'Т',
    'X': 'Х', 'C': 'С', 'E': 'Е', 'Y': 'У',
    'b': 'В', 'o': 'О', 'p': 'Р', 'a': 'А',
    'h': 'Н', 'k': 'К', 'm': 'М', 't': 'Т',
    'x': 'Х', 'c': 'С', 'e': 'Е', 'y': 'У',
    '?': '', ' ': '',
}
_TRANS_TABLE = str.maketrans(_CORRECTIONS)

def _apply_corrections(text: str) -> str:
    """Применяет коррекции и удаляет любой мусор, оставляя только кириллицу и цифры."""
    text = text.translate(_TRANS_TABLE)
    # Оставляем только русские буквы и цифры, удаляя спецсимволы, пробелы и остаточную латиницу
    return re.sub(r'[^А-Я0-9]', '', text)


def get_reader():
    global _READER
    if _READER is None:
        try:
            logger.info("Загрузка EasyOCR...")
            _READER = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
            logger.info("✅ EasyOCR загружен")
        except Exception as e:
            logger.error(f"❌ Ошибка EasyOCR: {e}")
            _READER = False
    return _READER if _READER is not False else None


def _apply_corrections(text: str) -> str:
    """Применяет коррекции символов через translate (O(n) вместо O(n*k))."""
    return text.translate(_TRANS_TABLE)


def preprocess_plate_image(plate_roi: np.ndarray) -> np.ndarray:
    if plate_roi is None or plate_roi.size == 0:
        raise ValueError("Пустое изображение")


    # Переводим в оттенки серого
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if len(plate_roi.shape) == 3 else plate_roi.copy()

    # Вместо бинаризации делаем легкое выравнивание гистограммы и размытие
    enhanced = _CLAHE.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Оставляем изображение серым — EasyOCR так работает значительно лучше!
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened



def recognize_plate_text(plate_roi: np.ndarray, allow_fallback: bool = True) -> Optional[str]:
    if plate_roi is None or plate_roi.size == 0:
        return None

    reader = get_reader()
    if reader is None:
        return None

    try:
        processed = preprocess_plate_image(plate_roi)
        results = reader.readtext(processed, detail=1, paragraph=False)

        if not results:
            results = reader.readtext(plate_roi, detail=1, paragraph=False)
            if not results:
                print("[OCR] Текст не распознан")
                return None

        raw_text = ''.join(r[1] for r in results).strip().upper()
        print(f"[OCR] Raw: '{raw_text}'")

        corrected_text = raw_text.translate(_TRANS_TABLE)
        print(f"[OCR] Corrected: '{corrected_text}'")

        # Оставляем только разрешенные символы
        cleaned = re.sub(r'[^АВЕКМНОРСТУХ0-9]', '', corrected_text)
        print(f"[OCR] Cleaned: '{cleaned}'")

        # --- ВАША ТЕКУЩАЯ СТРОКА ---
        corrected_text = raw_text.translate(_TRANS_TABLE)
        print(f"[OCR] Corrected: '{corrected_text}'")

        # --- ДОБАВЬТЕ ЭТОТ БЛОК ИСПРАВЛЕНИЯ ---
        def fix_a_o_confusion(text):
            # Если в тексте есть 'О', которая визуально похожа на 'А'
            # Заменяем только если структура номера предполагает букву на этом месте
            # Например, первая позиция или две последние
            if len(text) >= 6:
                chars = list(text)
                # Позиции 0, 4, 5 в стандартном номере - это буквы
                for i in [0, 4, 5]:
                    if i < len(chars) and chars[i] == 'О':
                        chars[i] = 'А'
                return "".join(chars)
            return text

        corrected_text = fix_a_o_confusion(corrected_text)
        print(f"[OCR] Fixed: '{corrected_text}'")
        # --------------------------------------

        # Оставляем только разрешенные символы
        cleaned = re.sub(r'[^АВЕКМНОРСТУХ0-9]', '', corrected_text)

        # 1. Пытаемся найти правильный паттерн: Буква + 3 цифры + 2 буквы
        # Номер может быть в любом месте строки
        patterns = [
            r'([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})',  # Стандартный
            r'([АВЕКМНОРСТУХ])(\d{2})([АВЕКМНОРСТУХ]{2})',  # 2 цифры (некоторые старые номера)
            r'([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ])',  # 1 буква в конце
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                if len(match.group(2)) == 3:
                    result = f"{match.group(1)}{match.group(2)}{match.group(3)}"
                elif len(match.group(2)) == 2:
                    # Добавляем недостающую цифру (часто 0 или 9 теряется)
                    result = f"{match.group(1)}0{match.group(2)}{match.group(3)}"
                else:
                    result = f"{match.group(1)}{match.group(2)}00{match.group(3)}"

                print(f"[OCR] ✅ Паттерн найден: {result}")
                return result

        # 2. Если паттерн не найден, пробуем "склеить" разорванные символы
        # Часто OCR путает '0' и 'О', '1' и 'I' и т.д.

        # Удаляем лишние символы в конце (RUS, RUSSIA и т.п.)
        cleaned = re.sub(r'(RUS|RUSSIA|RUSSIAN)$', '', cleaned)

        # Ищем первую букву, затем 3-4 цифры, затем 2 буквы
        match_long = re.search(r'([АВЕКМНОРСТУХ])(\d{3,4})([АВЕКМНОРСТУХ]{1,2})', cleaned)
        if match_long:
            letter1 = match_long.group(1)
            digits = match_long.group(2)[:3]  # Берем первые 3 цифры
            letters2 = match_long.group(3)[:2]  # Берем первые 2 буквы
            result = f"{letter1}{digits}{letters2}"
            print(f"[OCR] ✅ Извлечено из длинной строки: {result}")
            return result

        # 3. Берем первые 6 символов если есть
        if len(cleaned) >= 6:
            # Проверяем, что первый символ - буква
            first_char = cleaned[0]
            if first_char in 'АВЕКМНОРСТУХ':
                result = cleaned[:6]
                print(f"[OCR] ⚠️ Взяты первые 6 символов: {result}")
                return result

        print(f"[OCR] ❌ Не удалось извлечь номер из: '{cleaned}'")
        return None

    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        print(f"[OCR] Ошибка: {e}")
        return None


def smart_fix_text(text: str) -> str:
    """
    Принудительно меняет 'О' на 'А' в позициях, где должны быть буквы.
    Формат номера РФ: [Б][Ц][Ц][Ц][Б][Б]
    Индексы букв: 0, 4, 5
    """
    chars = list(text)
    letter_indices = [0, 4, 5]  # Позиции, где 100% должна быть буква

    for i in letter_indices:
        if i < len(chars) and chars[i] == 'О':
            chars[i] = 'А'
    return "".join(chars)


def _alternative_recognition(plate_roi: np.ndarray, reader) -> Optional[str]:
    try:
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if len(plate_roi.shape) == 3 else plate_roi.copy()

        h, w = gray.shape
        if h < 100 or w < 300:
            scale = max(2.0, 300.0 / w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results = reader.readtext(binary, detail=0, paragraph=False)

        if results:
            return _apply_corrections(results[0].strip().upper())  # Переиспользуем общую функцию

    except Exception as e:
        logger.error(f"Альтернативное распознавание не удалось: {e}")
    return None


def validate_russian_plate(text: str) -> bool:
    if not text or len(text) < 6:
        return False
    return bool(_PATTERN_VALID.match(text)) or bool(_PATTERN_VALID_REGION.match(text))


def recognize_with_validation(plate_roi: np.ndarray) -> Optional[str]:
    text = recognize_plate_text(plate_roi, allow_fallback=False)
    return text if text and validate_russian_plate(text) else None


if __name__ == "__main__":
    test_img = np.ones((100, 400, 3), dtype=np.uint8) * 245
    cv2.rectangle(test_img, (0, 0), (400, 100), (200, 200, 200), -1)
    cv2.putText(test_img, "А721АВ", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.rectangle(test_img, (45, 25), (300, 80), (0, 0, 0), 2)

    result = recognize_plate_text(test_img)
    print(f"\nРезультат: '{result}'")
    print(f"Валидация 'А721АВ': {validate_russian_plate('А721АВ')}")
    print(f"Валидация 'A721AB': {validate_russian_plate('A721AB')}")