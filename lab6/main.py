import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Генерация выборок
np.random.seed(42)
n = 1000
X1 = np.random.normal(0, 0.95, n)
X2 = np.random.normal(1, 1.05, n)


# 2. Функции для вычисления внутренних и внешних оценок
def get_im_out(X):
    Q1, Q3 = np.percentile(X, [25, 75])
    X_min, X_max = np.min(X), np.max(X)
    # print(Q1, Q3, X_min, X_max)
    return {'Im': [Q1, Q3], 'Out': [X_min, X_max]}


# 3. Вычисление оценок для исходных выборок
X1_im, X1_out = get_im_out(X1)['Im'], get_im_out(X1)['Out']
X2_im, X2_out = get_im_out(X2)['Im'], get_im_out(X2)['Out']
print("x1 ценки: ", X1_im, X1_out)
print("x2 ценки: ", X2_im, X2_out)


# 4. Функция для вычисления индекса Жаккара
def jaccard_index(interval1, interval2):
    a1, b1 = interval1
    a2, b2 = interval2

    # Пересечение
    intersection = max(0, min(b1, b2) - max(a1, a2))
    if intersection <= 0:
        return 0

    # Объединение
    union = max(b1, b2) - min(a1, a2)

    return intersection / union


# 5. Поиск оптимального параметра сдвига
a_values = np.linspace(-3, 3, 500)
J_im = []
J_out = []

for a in a_values:
    X1_shifted = X1 + a

    # Вычисляем оценки для сдвинутой выборки
    X1s_im, X1s_out = get_im_out(X1_shifted)['Im'], get_im_out(X1_shifted)['Out']

    # Вычисляем индексы Жаккара
    J_im.append(jaccard_index(X1s_im, X2_im))
    J_out.append(jaccard_index(X1s_out, X2_out))

# 6. Находим оптимальные значения параметра сдвига
a_im = a_values[np.argmax(J_im)]
a_out = a_values[np.argmax(J_out)]

# 7. Визуализация результатов
plt.figure(figsize=(12, 6))

plt.plot(a_values, J_im, label=f'J_Im (max at a={a_im:.2f})', color='blue')
plt.plot(a_values, J_out, label=f'J_Out (max at a={a_out:.2f})', color='red')
plt.axvline(x=1, color='green', linestyle='--', label='Истинный сдвиг (a=1)')
plt.xlabel('Параметр сдвига a')
plt.ylabel('Индекс Жаккара J')
plt.title('Зависимость индексов Жаккара от параметра сдвига')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 8. Вывод результатов
print("Результаты:")
print(f"Оценка параметра сдвига по внутренним оценкам (Im): a_Im = {a_im:.4f}")
print(f"Оценка параметра сдвига по внешним оценкам (Out): a_Out = {a_out:.4f}")
print(f"Истинное значение сдвига: 1.0")