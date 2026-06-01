import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from pathlib import Path
import os

# --- ⚙️ Настройки проекта ---
DATA_DIR = r'D:\Test1\color_dataset'  # r-строка для Windows-путей
IMG_SIZE = (224, 224)  # Нативный размер MobileNetV2
BATCH_SIZE = 32
INITIAL_EPOCHS = 30   # Первая фаза (замороженная база)
FINE_TUNE_EPOCHS = 50 # Вторая фаза (fine-tuning)
NUM_CLASSES = None    # Определится автоматически

# --- 🚀 Подготовка данных с улучшенной аугментацией ---
print("--- Загрузка и подготовка данных...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,           # ✅ Ваше улучшение
    brightness_range=[0.8, 1.2],    # ✅ Ваше улучшение
    channel_shift_range=30,         # ✅ Ваше улучшение
    fill_mode='nearest'
)

# Генератор данных для обучения
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Генератор для валидации (без аугментации)
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Найдено классов: {NUM_CLASSES}")
print(f"Классы: {train_generator.class_indices}")

# --- 🛠️ Создание модели ---
print("--- Построение CNN-модели...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# ---- ФАЗА 1: Замороженная база ----
base_model.trainable = False

# Голова классификатора
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)  # Дополнительный слой
x = layers.Dropout(0.2)(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=predictions)

# Компиляция для первой фазы (стандартный learning rate)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 🏋️ Фаза 1: Обучение с замороженной базой ---
print("--- ФАЗА 1: Обучение замороженной модели...")

callbacks_phase1 = [
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1)
]

history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    callbacks=callbacks_phase1,
    verbose=1
)

# --- 🔧 Фаза 2: Fine-tuning ---
print("--- ФАЗА 2: Fine-tuning (разморозка верхних слоев)...")

# Размораживаем только верхние 30 слоев
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Перекомпилируем с уменьшенным learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # ✅ Ваше улучшение
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=1e-6),
    EarlyStopping(patience=7, restore_best_weights=True, verbose=1)
]

# Продолжаем обучение
history2 = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    callbacks=callbacks_phase2,
    verbose=1
)

# --- 💾 Сохранение модели ---
MODEL_SAVE_PATH = 'trained_color_classifier_finetuned.h5'
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Модель успешно обучена и сохранена в {MODEL_SAVE_PATH}")

# --- 📊 Сохранение истории обучения ---
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump({'phase1': history1.history, 'phase2': history2.history}, f)
print("📈 История обучения сохранена в training_history.pkl")