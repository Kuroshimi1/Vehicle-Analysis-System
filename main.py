from __future__ import annotations
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
import threading
import time
import sqlite3
import queue
import os
import re
from PIL import Image, ImageDraw, ImageFont, ImageTk

from stubs.car_detector import detect_cars
from stubs.plate_detector import detect_plates
from stubs.color_classifier import classify_color
from stubs.ocr_recognizer import recognize_plate_text

TARGET_RESOLUTION = (1280, 720)
SAMPLING_INTERVAL_SECONDS = 0.3

# Мягкая, благородная темная палитра (без "кислоты")
BG_MAIN = "#0B131F"  # Глубокий полуночный синий (спокойный фон)
BG_CARD = "#142132"  # Чуть более светлый тон для карточек и таблиц
ACCENT_CYAN = "#48CAE4"  # Мягкий бирюзово-голубой для акцентов
TEXT_MAIN = "#E0E6ED"  # Мягкий белый (не режет глаза)
TEXT_MUTED = "#7E8B9B"  # Приглушенный серый для второстепенного текста
BUTTON_BG = "#1D2D44"  # Спокойный цвет кнопок
BUTTON_ACTIVE = "#0096C7"  # Цвет при наведении


def apply_dark_theme(style: ttk.Style):
    """Настройка аккуратного и мягкого темного интерфейса"""
    style.theme_use("clam")

    # Сброс базовых параметров
    style.configure(".", background=BG_MAIN, foreground=TEXT_MAIN, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG_MAIN)
    style.configure("TLabel", background=BG_MAIN, foreground=TEXT_MAIN)

    # Заголовки (крупные, но не кричащие)
    style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"), foreground=TEXT_MAIN, background=BG_MAIN)
    style.configure("Subheader.TLabel", font=("Segoe UI", 10), foreground=TEXT_MUTED, background=BG_MAIN)

    # Кнопки: плоские, без жестких рамок
    style.configure("TButton", background=BUTTON_BG, foreground=TEXT_MAIN, borderwidth=0,
                    font=("Segoe UI", 10, "bold"), padding=(12, 6))
    style.map("TButton",
              background=[("active", BUTTON_ACTIVE), ("disabled", "#151F2E")],
              foreground=[("active", "#FFFFFF"), ("disabled", TEXT_MUTED)])

    # Слайдер и Прогресс-бар
    style.configure("Horizontal.TScale", background=BG_MAIN, troughcolor=BG_CARD, sliderlength=14, borderwidth=0)
    style.map("Horizontal.TScale", background=[("active", ACCENT_CYAN)])

    style.configure("Horizontal.TProgressbar", troughcolor=BG_CARD, background=ACCENT_CYAN, thickness=6, borderwidth=0)

    # Современные таблицы (без сетки "в клеточку")
    style.configure("Treeview", background=BG_CARD, fieldbackground=BG_CARD, foreground=TEXT_MAIN,
                    rowheight=28, font=("Segoe UI", 10), borderwidth=0)
    style.configure("Treeview.Heading", background=BUTTON_BG, foreground=TEXT_MAIN,
                    font=("Segoe UI", 10, "bold"), borderwidth=0, padding=5)
    style.map("Treeview", background=[("selected", BUTTON_ACTIVE)], foreground=[("selected", "#FFFFFF")])

    # Поля ввода
    style.configure("TEntry", fieldbackground=BG_CARD, foreground=TEXT_MAIN, borderwidth=1, bordercolor=BUTTON_BG)
    style.configure("TCombobox", fieldbackground=BG_CARD, foreground=TEXT_MAIN, arrowcolor=ACCENT_CYAN, borderwidth=0)
    style.map("TCombobox", fieldbackground=[("readonly", BG_CARD)], foreground=[("readonly", TEXT_MAIN)])

    # Карточки группировки
    style.configure("TLabelframe", background=BG_MAIN, bordercolor=BG_CARD, borderwidth=1)
    style.configure("TLabelframe.Label", background=BG_MAIN, foreground=ACCENT_CYAN, font=("Segoe UI", 10, "bold"))


class VehicleTracker:
    def __init__(self):
        self.vehicle_map = {}
        self.next_car_id = 1
        self.pending_records = []
        self._lock = threading.Lock()

    def get_or_create_car_id(self, plate_text, color):
        key = (plate_text or "UNKNOWN", color or "UNCLASSIFIED")
        with self._lock:
            if key not in self.vehicle_map:
                car_id = f"CAR_{self.next_car_id}"
                self.vehicle_map[key] = car_id
                self.next_car_id += 1
            return self.vehicle_map[key]

    def add_record(self, frame_timestamp, detections):
        with self._lock:
            for d in detections:
                self.pending_records.append({
                    'frame_id': frame_timestamp,
                    'car_id': d[0], 'plate_text': d[1],
                    'color': d[2], 'color_confidence': d[3]
                })

    def get_pending_records(self):
        with self._lock:
            records = self.pending_records[:]
            self.pending_records = []
        return records


class DatabaseManager:
    def __init__(self, db_name="vehicle_analysis.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._lock = threading.Lock()
        self._setup_db()

    def _setup_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                frame_id INTEGER, car_id TEXT, plate_text TEXT,
                color TEXT, color_confidence REAL,
                UNIQUE(plate_text, color) ON CONFLICT REPLACE
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS wanted_list (
                plate_text TEXT PRIMARY KEY,
                color TEXT,
                reason TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def write_records(self, records):
        if not records:
            return
        TIME_WINDOW = 3600
        grouped = {}
        for r in records:
            grouped.setdefault((r['plate_text'], r['color']), []).append(r)

        final = []
        with self._lock:
            for (plate, color), rlist in grouped.items():
                rlist.sort(key=lambda r: r['frame_id'])
                earliest = rlist[0]
                self.cursor.execute(
                    "SELECT 1 FROM vehicle_detections WHERE plate_text=? AND color=? AND frame_id>=?",
                    (plate, color, earliest['frame_id'] - TIME_WINDOW)
                )
                if not self.cursor.fetchone():
                    final.append(earliest)
            if final:
                self.cursor.executemany(
                    "INSERT INTO vehicle_detections VALUES (?,?,?,?,?)",
                    [(r['frame_id'], r['car_id'], r['plate_text'], r['color'], r['color_confidence']) for r in final]
                )
                self.conn.commit()
        print(f"DB: inserted {len(final)} records.")

    def check_wanted_status(self, plate_text, detected_color_en):
        if not plate_text:
            return False, None

        color_translation = {
            "BLACK": "Черный", "BLUE": "Синий", "BROWN": "Коричневый",
            "GREEN": "Зеленый", "GRAY": "Серый", "SILVER": "Серый",
            "ORANGE": "Оранжевый", "PINK": "Розовый", "PURPLE": "Фиолетовый",
            "RED": "Красный", "WHITE": "Белый", "YELLOW": "Желтый"
        }

        detected_color_ru = color_translation.get(detected_color_en.upper(), detected_color_en)

        with self._lock:
            self.cursor.execute("SELECT color, reason FROM wanted_list WHERE plate_text = ?", (plate_text,))
            row = self.cursor.fetchone()

            if row:
                wanted_color, reason = row
                if wanted_color is None or wanted_color == "":
                    return True, f"{reason} (Розыск по номеру)"

                if wanted_color.strip().lower() == detected_color_ru.strip().lower():
                    return True, f"{reason} (Совпадение по номеру и цвету)"

        return False, None


class VideoAnalysisApp:
    def open_db_viewer(self):
        DatabaseViewerWindow(self)

    def __init__(self, master):
        self.master = master
        master.title("Real-Time Vehicle Analysis System")
        master.configure(bg=BG_MAIN)

        self.style = ttk.Style()
        apply_dark_theme(self.style)

        self.tracker = VehicleTracker()
        self.db_manager = DatabaseManager()

        self.analysis_results = {}
        self.analysis_done = False
        self.analysis_progress = 0

        self.video_path = None
        self.is_playing = False
        self.playback_thread = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 25

        self.alerted_vehicles = set()
        self.font = self._load_font(14)
        self._setup_gui()

    def _load_font(self, size):
        for path in [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except:
                    pass
        return ImageFont.load_default()

    def _setup_gui(self):
        # Контейнер заголовков с хорошими отступами
        header_frame = ttk.Frame(self.master)
        header_frame.pack(anchor="w", padx=30, pady=(25, 15))

        ttk.Label(header_frame, text="Vehicle Analysis System", style="Header.TLabel").pack(anchor="w")


        # Окно вывода видео с очень аккуратной, едва заметной рамкой
        video_border = tk.Frame(self.master, bg=BG_CARD, bd=1)
        video_border.pack(padx=30, pady=10)

        self.canvas = tk.Canvas(video_border, width=TARGET_RESOLUTION[0], height=TARGET_RESOLUTION[1], bg='#080F18',
                                highlightthickness=0)
        self.canvas.pack()

        # Элементы управления видео
        controls_frame = ttk.Frame(self.master)
        controls_frame.pack(fill='x', padx=30, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, maximum=100,
                                            style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill='x', pady=5)

        self.seek_var = tk.IntVar()
        self.seek_slider = ttk.Scale(controls_frame, from_=0, to=100, orient='horizontal', variable=self.seek_var,
                                     command=self._on_seek, style="Horizontal.TScale")
        self.seek_slider.pack(fill='x', pady=2)

        self.status_label = tk.Label(self.master, text="📂 Откройте видеофайл для начала работы системы", fg=TEXT_MUTED,
                                     bg=BG_MAIN, font=("Segoe UI", 10))
        self.status_label.pack(anchor="w", padx=30, pady=5)

        # Панель кнопок (кнопки стали аккуратнее, без рамок)
        btn_frame = ttk.Frame(self.master)
        btn_frame.pack(anchor="w", padx=30, pady=(15, 30))

        ttk.Button(btn_frame, text="📂 Открыть видео", command=self.open_video).pack(side='left', padx=4)
        self.play_btn = ttk.Button(btn_frame, text="▶ Воспроизвести", command=self.toggle_playback, state='disabled')
        self.play_btn.pack(side='left', padx=4)
        ttk.Button(btn_frame, text="💾 Сохранить в БД", command=self.trigger_db_write).pack(side='left', padx=4)
        ttk.Button(btn_frame, text="📊 Панель управления БД", command=self.open_db_viewer).pack(side='left', padx=4)

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=(("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*"))
        )
        if path:
            dir_name = os.path.dirname(path)
            base_name = os.path.basename(path)
            redacted_dir = os.path.join(dir_name, '.idea', 'redacted')

            name_without_ext, ext = os.path.splitext(base_name)
            redacted_filename = name_without_ext + '.redacted' + ext
            redacted_path = os.path.join(redacted_dir, redacted_filename)

            if os.path.exists(redacted_path):
                path = redacted_path

            self.video_path = path
            self.alerted_vehicles.clear()
            self._start_analysis(path)

    def _start_analysis(self, path):
        self.analysis_results = {}
        self.analysis_done = False
        self.is_playing = False
        self.play_btn.config(state='disabled', text="▶ Воспроизвести")
        self.status_label.config(text="⏳ Выполняется нейросетевой анализ структуры видеопотока...", fg=ACCENT_CYAN)

        threading.Thread(target=self._analysis_worker, args=(path,), daemon=True).start()

    def _analysis_worker(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.master.after(0, lambda: self.status_label.config(text="Ошибка открытия файла", fg='#EF4444'))
            return

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_ms = self.total_frames / self.fps * 1000
        self.seek_slider.config(to=self.total_frames - 1)

        sample_interval_ms = SAMPLING_INTERVAL_SECONDS * 1000
        next_sample_ms = 0.0
        processed = 0

        plate_pattern = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}$')

        print(f"\n{'=' * 60}")
        print(f"🚗 НАЧАЛО АНАЛИЗА ВИДЕО")
        print(f"{'=' * 60}")
        print(f"Файл: {path}")
        print(f"Всего кадров: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Интервал анализа: {SAMPLING_INTERVAL_SECONDS} сек")
        print(f"{'=' * 60}\n")

        while next_sample_ms <= total_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, next_sample_ms)
            ret, frame = cap.read()
            if not ret:
                break

            actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            print(f"\n📸 Кадр {frame_idx} (время: {actual_ms / 1000:.1f} сек)")

            bboxes = detect_cars(frame)
            print(f"   Обнаружено автомобилей: {len(bboxes)}")

            detections = []
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                car_roi = frame[y1:y2, x1:x2]
                print(f"   🚗 Авто {idx + 1}: позиция [{x1},{y1},{x2},{y2}]")

                plate_bbox = detect_plates(car_roi)
                plate_text = None
                if plate_bbox:
                    px1, py1, px2, py2 = plate_bbox
                    print(f"      Найден кандидат в номер: [{px1},{py1},{px2},{py2}]")
                    raw_text = recognize_plate_text(car_roi[py1:py2, px1:px2])
                    if raw_text:
                        cleaned_text = raw_text.strip().replace(" ", "").upper()
                        if plate_pattern.match(cleaned_text):
                            plate_text = cleaned_text
                            print(f"      ✅ Номер распознан: {plate_text}")
                        else:
                            print(f"      ❌ Некорректный формат номера: {cleaned_text}")
                    else:
                        print(f"      ❌ Номер не распознан OCR")
                else:
                    print(f"      ⚠️ Номерной знак не обнаружен")

                color, color_conf = classify_color(car_roi)
                print(f"      🎨 Цвет: {color} (уверенность: {color_conf:.2f})")

                car_id = self.tracker.get_or_create_car_id(plate_text, color)
                print(f"      🆔 ID автомобиля: {car_id}")

                detections.append((car_id, plate_text, color, color_conf))

            self.analysis_results[frame_idx] = {
                'bboxes': bboxes,
                'detections': detections,
                'timestamp': int(actual_ms / 1000)
            }

            self.tracker.add_record(int(actual_ms / 1000), detections)

            processed += 1
            progress = min(100.0, (next_sample_ms / total_ms) * 100) if total_ms > 0 else 100
            self.master.after(0, self._update_progress, progress)
            next_sample_ms += sample_interval_ms

        cap.release()
        self.analysis_done = True
        self.current_frame_idx = 0
        self.master.after(0, self._on_analysis_complete)

        print(f"\n{'=' * 60}")
        print(f"✅ АНАЛИЗ ЗАВЕРШЕН")
        print(f"Обработано кадров: {len(self.analysis_results)}")
        print(f"{'=' * 60}\n")

    def _update_progress(self, value):
        self.progress_var.set(value)

    def _on_analysis_complete(self):
        self.status_label.config(
            text=f"✅ Анализ завершён. Успешно обработано {len(self.analysis_results)} ключевых кадров.", fg=ACCENT_CYAN)
        self.play_btn.config(state='normal')

    def toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Воспроизвести")
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Пауза")
            if self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
                self.playback_thread.start()

    def _playback_worker(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        frame_duration = 1.0 / self.fps
        sample_keys = sorted(self.analysis_results.keys())

        while self.is_playing:
            frame_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = cv2.VideoCapture(self.video_path)
                self.current_frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    break

            # Получаем результаты анализа для текущего кадра
            result = self._get_nearest_result(sample_keys, self.current_frame_idx)
            if result:
                frame = self._draw_detections(frame, result['bboxes'], result['detections'])
                # Проверяем розыск в отдельном потоке
                threading.Thread(target=self._check_realtime_wanted, args=(result['detections'],), daemon=True).start()

            # Быстрая конвертация кадра
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = min(TARGET_RESOLUTION[0] / w, TARGET_RESOLUTION[1] / h)

            if scale != 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                # ИСПРАВЛЕНО: используем cv2.INTER_LINEAR вместо INTER_FAST
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(img)

            # Обновляем GUI в основном потоке
            self.master.after(0, self._update_canvas, imgtk, self.current_frame_idx)

            self.current_frame_idx += 1

            # Точная задержка для поддержания FPS
            elapsed = time.perf_counter() - frame_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    def _update_canvas(self, imgtk, frame_idx):
        """Быстрое обновление canvas"""
        # Удаляем старую картинку, если есть
        if hasattr(self.canvas, 'current_image'):
            self.canvas.delete(self.canvas.current_image)

        # Создаем новую картинку
        self.canvas.current_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # сохраняем ссылку

        # Обновляем слайдер не на каждом кадре для экономии ресурсов
        current_val = self.seek_var.get()
        if abs(current_val - frame_idx) > 5 or frame_idx % 10 == 0:
            self.seek_var.set(frame_idx)

    def _check_realtime_wanted(self, detections):
        for car_id, plate_text, color, _ in detections:
            if not plate_text:
                continue

            alert_key = (plate_text, car_id)
            if alert_key in self.alerted_vehicles:
                continue

            is_wanted, reason = self.db_manager.check_wanted_status(plate_text, color)
            if is_wanted:
                self.alerted_vehicles.add(alert_key)
                self.master.after(0, self._show_wanted_alert, plate_text, color, reason)

    def _show_wanted_alert(self, plate, color, reason):
        if self.is_playing:
            self.toggle_playback()

        messagebox.showwarning(
            "ВНИМАНИЕ! ОБНАРУЖЕН РОЗЫСК",
            f"Зафиксирован автомобиль из списка розыска!\n\n"
            f"Гос. номер: {plate}\n"
            f"Распознанный цвет: {color}\n"
            f"Причина: {reason}",
            parent=self.master
        )

    def _get_nearest_result(self, sample_keys, frame_idx):
        if not sample_keys:
            return None
        lo, hi = 0, len(sample_keys) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sample_keys[mid] <= frame_idx:
                best = sample_keys[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return self.analysis_results.get(best)

    def _display_frame(self, frame, frame_idx):
        h, w = frame.shape[:2]
        scale = min(TARGET_RESOLUTION[0] / w, TARGET_RESOLUTION[1] / h)
        if scale != 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        self.seek_var.set(frame_idx)

    def _on_seek(self, val):
        self.current_frame_idx = int(float(val))

    def _draw_detections(self, frame, bboxes, detections):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        for bbox, det in zip(bboxes, detections):
            x1, y1, x2, y2 = bbox
            # Распаковываем все 4 элемента правильно
            car_id, plate_text, color, color_conf = det  # <-- ИСПРАВЛЕНО

            is_wanted, _ = self.db_manager.check_wanted_status(plate_text, color)

            # Аккуратные цвета обводки на видеопотоке
            box_color = (239, 68, 68) if is_wanted else (72, 202, 228)
            text_bg_color = (153, 27, 27) if is_wanted else (20, 33, 50)

            # Используем plate_text
            display_plate = plate_text if plate_text else "???"
            label = f"ID:{car_id} | {display_plate} | {color}"
            if is_wanted:
                label += " [РОЗЫСК]"

            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            tb = draw.textbbox((x1, y1 - 22), label, font=self.font)
            draw.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=text_bg_color)
            draw.text((x1 + 2, y1 - 22), label, fill=(240, 244, 248), font=self.font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def trigger_db_write(self):
        records = self.tracker.get_pending_records()
        threading.Thread(target=self.db_manager.write_records, args=(records,), daemon=True).start()


class DataEditDialog(tk.Toplevel):
    def __init__(self, parent, title="Запись", initial_values=None):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG_MAIN)
        self.result = None

        self.transient(parent)
        self.grab_set()
        self.focus_set()

        self.fields = ["Frame ID", "Car ID", "Plate Text", "Color", "Conf"]
        self.entries = {}

        card = ttk.Frame(self, padding=20)
        card.pack(fill="both", expand=True, padx=20, pady=20)

        for i, field in enumerate(self.fields):
            ttk.Label(card, text=f"{field}:").grid(row=i, column=0, padx=10, pady=6, sticky="e")
            entry = ttk.Entry(card, font=("Segoe UI", 10))
            entry.grid(row=i, column=1, padx=10, pady=6, sticky="w")

            if initial_values and i < len(initial_values):
                val = initial_values[i]
                if field == "Plate Text" and val == "None":
                    val = ""
                entry.insert(0, str(val))
            elif field == "Conf" and not initial_values:
                entry.insert(0, "1.0")

            self.entries[field] = entry

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(0, 20))

        ttk.Button(btn_frame, text="Сохранить", command=self.on_save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side="left", padx=5)

        self.center_window()

    def center_window(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() - self.winfo_width()) // 2
        y = self.master.winfo_y() + (self.master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def on_save(self):
        try:
            frame_id = int(self.entries["Frame ID"].get())
            car_id = self.entries["Car ID"].get().strip()
            raw_plate = self.entries["Plate Text"].get().strip().upper()
            color = self.entries["Color"].get().strip()
            conf = float(self.entries["Conf"].get())

            if not car_id or not color:
                raise ValueError("Поля Car ID и Color не могут быть пустыми.")

            plate_pattern = re.compile(r'^[А-ЯA-Z]\d{3}[А-ЯA-Z]{2}$')
            if raw_plate == "" or raw_plate == "NONE":
                plate_text = None
            elif plate_pattern.match(raw_plate):
                plate_text = raw_plate
            else:
                raise ValueError("Номер должен соответствовать формату Х000ХХ или оставаться пустым.")

            self.result = (frame_id, car_id, plate_text, color, conf)
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные данные: {e}", parent=self)


class DatabaseViewerWindow(tk.Toplevel):
    def __init__(self, master_app: VideoAnalysisApp):
        super().__init__(master_app.master)
        self.title("Управление Базой Данных Анализа")
        self.master_app = master_app
        self.geometry("860x520")
        self.configure(bg=BG_MAIN)

        self.transient(master_app.master)
        self.focus_set()

        title_frame = ttk.Frame(self, padding=15)
        title_frame.pack(fill="x", padx=10)
        ttk.Label(title_frame, text="Архив детекций ТС", font=("Segoe UI", 14, "bold"), foreground=ACCENT_CYAN).pack(
            anchor="w")

        frame = ttk.Frame(self)
        frame.pack(padx=20, pady=5, fill="both", expand=True)

        self.columns = ("Frame ID", "Car ID", "Plate Text", "Color", "Conf")
        self.tree = ttk.Treeview(frame, columns=self.columns, show='headings', selectmode="extended")

        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        self.tree.pack(fill="both", expand=True)

        crud_frame = ttk.Frame(self, padding=20)
        crud_frame.pack(fill="x")

        ttk.Button(crud_frame, text="➕ Добавить запись", command=self.add_record).pack(side="left", padx=4)
        ttk.Button(crud_frame, text="✏️ Изменить", command=self.edit_record).pack(side="left", padx=4)
        ttk.Button(crud_frame, text="❌ Удалить", command=self.delete_record).pack(side="left", padx=4)

        ttk.Button(crud_frame, text="📋 Открыть базу розыска", command=self.open_wanted_list).pack(side="left", padx=25)
        ttk.Button(crud_frame, text="🔄 Обновить таблицу", command=self.load_data).pack(side="right", padx=4)

        self.load_data()

    def open_wanted_list(self):
        WantedListWindow(self, self.master_app.db_manager)

    def load_data(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            cursor = self.master_app.db_manager.conn.cursor()
            cursor.execute(
                "SELECT frame_id, car_id, plate_text, color, color_confidence FROM vehicle_detections ORDER BY frame_id DESC")
            rows = cursor.fetchall()

            for row in rows:
                display_row = list(row)
                if display_row[2] is None:
                    display_row[2] = "None"
                self.tree.insert("", "end", values=display_row)

        except sqlite3.Error as e:
            messagebox.showerror("Ошибка БД", f"Не удалось загрузить данные: {e}", parent=self)

    def add_record(self):
        dialog = DataEditDialog(self, title="Добавление новой записи")
        self.wait_window(dialog)

        if dialog.result:
            try:
                cursor = self.master_app.db_manager.conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO vehicle_detections (frame_id, car_id, plate_text, color, color_confidence) VALUES (?, ?, ?, ?, ?)",
                    dialog.result
                )
                self.master_app.db_manager.conn.commit()
                self.load_data()
            except sqlite3.Error as e:
                messagebox.showerror("Ошибка БД", f"Не удалось сохранить запись: {e}", parent=self)

    def edit_record(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Внимание", "Пожалуйста, выберите запись для редактирования.", parent=self)
            return

        current_values = self.tree.item(selected_items[0], "values")
        old_plate = None if current_values[2] == "None" else current_values[2]
        old_color = current_values[3]

        dialog = DataEditDialog(self, title="Редактирование записи", initial_values=current_values)
        self.wait_window(dialog)

        if dialog.result:
            try:
                cursor = self.master_app.db_manager.conn.cursor()
                cursor.execute(
                    """UPDATE vehicle_detections 
                       SET frame_id=?, car_id=?, plate_text=?, color=?, color_confidence=? 
                       WHERE plate_text IS ? AND color=?""",
                    (*dialog.result, old_plate, old_color)
                )
                self.master_app.db_manager.conn.commit()
                self.load_data()
            except sqlite3.Error as e:
                messagebox.showerror("Ошибка БД", f"Не удалось обновить запись: {e}", parent=self)

    def delete_record(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Внимание", "Пожалуйста, выберите записи для удаления.", parent=self)
            return

        count = len(selected_items)
        prompt_text = f"Вы действительно хотите удалить {count} выделенных записей?" if count > 1 else "Вы действительно хотите удалить выбранную запись?"

        confirm = messagebox.askyesno("Подтверждение", prompt_text, parent=self)
        if confirm:
            try:
                cursor = self.master_app.db_manager.conn.cursor()
                for item in selected_items:
                    current_values = self.tree.item(item, "values")
                    plate_text = None if current_values[2] == "None" else current_values[2]
                    color = current_values[3]

                    cursor.execute(
                        "DELETE FROM vehicle_detections WHERE plate_text IS ? AND color=?",
                        (plate_text, color)
                    )

                self.master_app.db_manager.conn.commit()
                self.load_data()
            except sqlite3.Error as e:
                messagebox.showerror("Ошибка БД", f"Не удалось выполнить удаление: {e}", parent=self)


class WantedListWindow(tk.Toplevel):
    def __init__(self, parent, db_manager: DatabaseManager):
        super().__init__(parent)
        self.title("Список автомобилей в розыске")
        self.db_manager = db_manager
        self.geometry("740x600")  # Увеличил высоту
        self.configure(bg=BG_MAIN)
        self.minsize(600, 500)  # Минимальный размер, чтобы кнопки не пропадали

        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # Основной контейнер
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        # ========== ВЕРХНЯЯ ПАНЕЛЬ - добавление записи ==========
        input_frame = ttk.LabelFrame(main_frame, text=" Внести транспортное средство в базу розыска ", padding=15)
        input_frame.pack(padx=15, pady=(10, 5), fill="x")

        ttk.Label(input_frame, text="Гос. номер:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.plate_entry = ttk.Entry(input_frame, width=15, font=("Segoe UI", 10))
        self.plate_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(input_frame, text="(Шаблон: Х000ХХ)", foreground=TEXT_MUTED).grid(row=0, column=2, padx=5, pady=5,
                                                                                    sticky="w")

        ttk.Label(input_frame, text="Цвет кузова:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.color_combo = ttk.Combobox(input_frame, width=14, values=[
            "", "Черный", "Синий", "Коричневый", "Зеленый", "Серый",
            "Оранжевый", "Розовый", "Фиолетовый", "Красный", "Белый", "Желтый"
        ], state="readonly")
        self.color_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.color_combo.set("")

        ttk.Label(input_frame, text="Причина розыска:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.reason_entry = ttk.Entry(input_frame, width=40, font=("Segoe UI", 10))
        self.reason_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Button(input_frame, text="➕ Зафиксировать в базе", command=self.add_to_wanted).grid(row=3, column=1, pady=8,
                                                                                                sticky="w")

        # ========== СЕРЕДИНА - таблица ==========
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(padx=15, pady=5, fill="both", expand=True)

        self.columns = ("Plate", "Color", "Reason", "Date")
        self.tree = ttk.Treeview(table_frame, columns=self.columns, show='headings', selectmode="extended")

        self.tree.heading("Plate", text="Гос. номер")
        self.tree.heading("Color", text="Цвет кузова")
        self.tree.heading("Reason", text="Основание/Причина")
        self.tree.heading("Date", text="Дата добавления")

        self.tree.column("Plate", width=110, anchor="center")
        self.tree.column("Color", width=110, anchor="center")
        self.tree.column("Reason", width=280, anchor="w")
        self.tree.column("Date", width=150, anchor="center")

        # Скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.tree.pack(side='left', fill="both", expand=True)

        # ========== НИЖНЯЯ ПАНЕЛЬ - кнопки управления ==========
        action_frame = ttk.Frame(main_frame, padding=15)
        action_frame.pack(fill="x", side="bottom")  # Прижимаем к низу

        # Левая группа кнопок
        left_btn_frame = ttk.Frame(action_frame)
        left_btn_frame.pack(side="left")

        edit_btn = ttk.Button(left_btn_frame, text="✏️ Редактировать", command=self.edit_wanted_record)
        edit_btn.pack(side="left", padx=4)

        delete_btn = ttk.Button(left_btn_frame, text="❌ Удалить из розыска", command=self.delete_from_wanted)
        delete_btn.pack(side="left", padx=4)

        # Правая группа кнопок
        right_btn_frame = ttk.Frame(action_frame)
        right_btn_frame.pack(side="right")

        refresh_btn = ttk.Button(right_btn_frame, text="🔄 Обновить", command=self.load_wanted_data)
        refresh_btn.pack(side="right", padx=4)

        # Двойной клик для редактирования
        self.tree.bind("<Double-Button-1>", lambda event: self.edit_wanted_record())

        # Загружаем данные
        self.load_wanted_data()

    def load_wanted_data(self):
        """Загрузка данных из БД в таблицу"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            cursor = self.db_manager.conn.cursor()
            cursor.execute("SELECT plate_text, color, reason, added_at FROM wanted_list ORDER BY added_at DESC")
            for row in cursor.fetchall():
                display_row = list(row)
                if not display_row[1]:
                    display_row[1] = "Не указан"
                self.tree.insert("", "end", values=display_row)
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка БД", f"Не удалось загрузить базу розыска: {e}", parent=self)

    def add_to_wanted(self):
        """Добавление новой записи в розыск"""
        plate = self.plate_entry.get().strip().upper()
        color = self.color_combo.get().strip()
        reason = self.reason_entry.get().strip()

        if not plate or not reason:
            messagebox.showwarning("Внимание", "Заполните поля Гос. номер и Причина розыска!", parent=self)
            return

        plate_pattern = re.compile(r'^[А-ЯA-Z]\d{3}[А-ЯA-Z]{2}$')
        if not plate_pattern.match(plate):
            messagebox.showerror("Ошибка", "Номер не соответствует стандарту Х000ХХ!\nПример: А123ВС", parent=self)
            return

        db_color = color if color != "" else None

        try:
            cursor = self.db_manager.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO wanted_list (plate_text, color, reason) VALUES (?, ?, ?)",
                (plate, db_color, reason)
            )
            self.db_manager.conn.commit()

            # Очищаем поля
            self.plate_entry.delete(0, tk.END)
            self.reason_entry.delete(0, tk.END)
            self.color_combo.set("")

            self.load_wanted_data()
            info_color = color if color else "не указан"
            messagebox.showinfo("Успех", f"Автомобиль {plate} (цвет: {info_color}) успешно занесен в розыск.",
                                parent=self)
        except sqlite3.Error as e:
            messagebox.showerror("Ошибка БД", f"Ошибка записи данных: {e}", parent=self)

    def edit_wanted_record(self):
        """Редактирование выбранной записи розыска"""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Внимание", "Пожалуйста, выберите запись для редактирования.", parent=self)
            return

        current_values = self.tree.item(selected_items[0], "values")
        old_plate = current_values[0]
        old_color = None if current_values[1] == "Не указан" else current_values[1]
        old_reason = current_values[2]

        # Диалог редактирования
        dialog = tk.Toplevel(self)
        dialog.title("Редактирование записи розыска")
        dialog.configure(bg=BG_MAIN)
        dialog.transient(self)
        dialog.grab_set()
        dialog.focus_set()
        dialog.resizable(False, False)

        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - 400) // 2
        y = self.winfo_y() + (self.winfo_height() - 250) // 2
        dialog.geometry(f"400x250+{x}+{y}")

        card = ttk.Frame(dialog, padding=20)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Гос. номер:").grid(row=0, column=0, padx=5, pady=8, sticky="e")
        plate_entry = ttk.Entry(card, width=15, font=("Segoe UI", 10))
        plate_entry.grid(row=0, column=1, padx=5, pady=8, sticky="w")
        plate_entry.insert(0, old_plate)
        ttk.Label(card, text="(Шаблон: Х000ХХ)", foreground=TEXT_MUTED).grid(row=0, column=2, padx=5, pady=8)

        ttk.Label(card, text="Цвет кузова:").grid(row=1, column=0, padx=5, pady=8, sticky="e")
        color_combo = ttk.Combobox(card, width=13, values=[
            "", "Черный", "Синий", "Коричневый", "Зеленый", "Серый",
            "Оранжевый", "Розовый", "Фиолетовый", "Красный", "Белый", "Желтый"
        ], state="readonly")
        color_combo.grid(row=1, column=1, padx=5, pady=8, sticky="w")
        color_combo.set(old_color if old_color else "")

        ttk.Label(card, text="Причина розыска:").grid(row=2, column=0, padx=5, pady=8, sticky="e")
        reason_entry = ttk.Entry(card, width=30, font=("Segoe UI", 10))
        reason_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=8, sticky="w")
        reason_entry.insert(0, old_reason)

        btn_frame = ttk.Frame(card)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=20)

        def save_changes():
            new_plate = plate_entry.get().strip().upper()
            new_color = color_combo.get().strip()
            new_reason = reason_entry.get().strip()

            if not new_plate or not new_reason:
                messagebox.showwarning("Внимание", "Заполните поля Гос. номер и Причина розыска!", parent=dialog)
                return

            plate_pattern = re.compile(r'^[А-ЯA-Z]\d{3}[А-ЯA-Z]{2}$')
            if not plate_pattern.match(new_plate):
                messagebox.showerror("Ошибка", "Номер не соответствует стандарту Х000ХХ!", parent=dialog)
                return

            db_color = new_color if new_color != "" else None

            try:
                cursor = self.db_manager.conn.cursor()
                cursor.execute("DELETE FROM wanted_list WHERE plate_text = ?", (old_plate,))
                cursor.execute(
                    "INSERT INTO wanted_list (plate_text, color, reason) VALUES (?, ?, ?)",
                    (new_plate, db_color, new_reason)
                )
                self.db_manager.conn.commit()

                self.load_wanted_data()
                dialog.destroy()
                messagebox.showinfo("Успех", f"Запись для {new_plate} успешно обновлена.", parent=self)
            except sqlite3.Error as e:
                messagebox.showerror("Ошибка БД", f"Не удалось обновить запись: {e}", parent=dialog)

        ttk.Button(btn_frame, text="💾 Сохранить", command=save_changes).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="❌ Отмена", command=dialog.destroy).pack(side="left", padx=5)

    def delete_from_wanted(self):
        """Удаление выбранных записей из розыска"""
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Внимание", "Выберите записи для удаления из списка розыска.", parent=self)
            return

        count = len(selected_items)
        confirm = messagebox.askyesno(
            "Подтверждение удаления",
            f"Вы действительно хотите удалить {count} запись(ей) из базы розыска?",
            parent=self
        )

        if confirm:
            try:
                cursor = self.db_manager.conn.cursor()
                deleted_plates = []

                for item in selected_items:
                    values = self.tree.item(item, "values")
                    plate = values[0]
                    cursor.execute("DELETE FROM wanted_list WHERE plate_text = ?", (plate,))
                    deleted_plates.append(plate)

                self.db_manager.conn.commit()
                self.load_wanted_data()

                messagebox.showinfo(
                    "Удаление выполнено",
                    f"Успешно удалено {len(deleted_plates)} записей из розыска.",
                    parent=self
                )
            except sqlite3.Error as e:
                messagebox.showerror("Ошибка БД", f"Не удалось выполнить удаление: {e}", parent=self)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Vehicle Analysis System")
    app = VideoAnalysisApp(root)
    root.mainloop()