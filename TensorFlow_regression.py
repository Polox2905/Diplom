import time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Начало замера времени
start_time = time.time()

# Генерация синтетических данных
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Получение предсказаний
y_pred = model.predict(X_test).flatten()

# Окончание замера времени
end_time = time.time()

# Оценка ошибки
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Time elapsed: {(end_time - start_time):.4f} seconds')

# Визуализация результатов
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, c='b', alpha=0.5, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('True Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Linear Regression Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.show()