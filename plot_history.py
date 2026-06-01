import pickle
import matplotlib.pyplot as plt

# 1. Загрузка данных
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

acc = history['phase1']['accuracy'] + history['phase2']['accuracy']
val_acc = history['phase1']['val_accuracy'] + history['phase2']['val_accuracy']
loss = history['phase1']['loss'] + history['phase2']['loss']
val_loss = history['phase1']['val_loss'] + history['phase2']['val_loss']

# 2. Создание графиков
plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(acc, label='Точность')
plt.plot(val_acc, label='Точность')
plt.axvline(x=len(history['phase1']['accuracy']), color='r', linestyle='--', label='Начало 2 фазы')
plt.title('Точность модели')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)

# График потерь
plt.subplot(1, 2, 2)
plt.plot(loss, label='Потери')
plt.plot(val_loss, label='Потери')
plt.axvline(x=len(history['phase1']['loss']), color='r', linestyle='--', label='Начало 2 фазы')
plt.title('Функция потерь')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()