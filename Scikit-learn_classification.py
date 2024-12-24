import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Начало замера времени
start_time = time.time()

# Загрузка данных
X, y = load_iris(return_X_y=True)

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Создание модели логистической регрессии
model = LogisticRegression()

# Обучение модели
model.fit(X_train, y_train)

# Предсказания на тестовом наборе
y_pred = model.predict(X_test)

# Окончание замера времени
end_time = time.time()

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)

# Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred)

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Вывод результатов
print(f'Accuracy: {accuracy:.4f}')
print(f'Time elapsed: {(end_time - start_time):.4f} seconds')