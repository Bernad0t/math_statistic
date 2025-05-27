import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Генерация данных
np.random.seed(42)
x = np.arange(-1.8, 2.1, 0.2)  # 20 точек с шагом 0.2 на [-1.8, 2]
n = len(x)

# Истинная зависимость
true_a, true_b = 2, 2
y_true = true_a + true_b * x

# Невозмущенная выборка
epsilon = np.random.normal(0, 1, n)
y_clean = y_true + epsilon

# Возмущенная выборка
y_noisy = y_clean.copy()
y_noisy[0] += 10  # y1 + 10
y_noisy[-1] += -10  # y20 - 10


# 2. Функции для оценки коэффициентов с соответствующими критериями

def least_squares(x, y):
    """Метод наименьших квадратов с критерием SSE"""
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a + b * x
    sse = np.sum((y - y_pred) ** 2)  # Сумма квадратов ошибок (SSE)
    return a, b, sse


def least_absolute(x, y):
    """Метод наименьших модулей с критерием SAE"""

    def objective(params):
        a, b = params
        return np.sum(np.abs(y - (a + b * x)))  # Сумма абсолютных ошибок (SAE)

    res = minimize(objective, [0, 0])
    a, b = res.x
    sae = objective([a, b])  # Значение критерия SAE
    return a, b, sae


# 3. Расчет коэффициентов и значений критериев

# Невозмущенная выборка
a_ls_clean, b_ls_clean, sse_clean = least_squares(x, y_clean)
a_la_clean, b_la_clean, sae_clean = least_absolute(x, y_clean)

# Возмущенная выборка
a_ls_noisy, b_ls_noisy, sse_noisy = least_squares(x, y_noisy)
a_la_noisy, b_la_noisy, sae_noisy = least_absolute(x, y_noisy)


# 4. Вывод результатов
def print_results(title, ls_results, la_results):
    print(f"\n{title}")
    print("-" * 85)
    print(f"{'Метод':<10} | {'â':<10} | {'Δa':<10} | {'b̂':<10} | {'Δb':<10} | {'Критерий':<15} | {'Значение':<10}")
    print("-" * 85)
    print(
        f"{'МНК':<10} | {ls_results[0]:<10.4f} | {ls_results[0] - true_a:<10.4f} | {ls_results[1]:<10.4f} | {ls_results[1] - true_b:<10.4f} | {'SSE':<15} | {ls_results[2]:<10.4f}")
    print(
        f"{'МНМ':<10} | {la_results[0]:<10.4f} | {la_results[0] - true_a:<10.4f} | {la_results[1]:<10.4f} | {la_results[1] - true_b:<10.4f} | {'SAE':<15} | {la_results[2]:<10.4f}")


print_results("Невозмущенная выборка",
              [a_ls_clean, b_ls_clean, sse_clean],
              [a_la_clean, b_la_clean, sae_clean])

print_results("Возмущенная выборка",
              [a_ls_noisy, b_ls_noisy, sse_noisy],
              [a_la_noisy, b_la_noisy, sae_noisy])

# 5. Визуализация
plt.figure(figsize=(14, 6))

# Невозмущенная выборка
plt.subplot(1, 2, 1)
plt.scatter(x, y_clean, label='Данные')
plt.plot(x, y_true, 'r-', label='Истинная зависимость')
plt.plot(x, a_ls_clean + b_ls_clean * x, 'g--', label=f'МНК (SSE={sse_clean:.2f})')
plt.plot(x, a_la_clean + b_la_clean * x, 'b--', label=f'МНМ (SAE={sae_clean:.2f})')
plt.title('Невозмущенная выборка')
plt.legend()

# Возмущенная выборка
plt.subplot(1, 2, 2)
plt.scatter(x, y_noisy, label='Данные')
plt.plot(x, y_true, 'r-', label='Истинная зависимость')
plt.plot(x, a_ls_noisy + b_ls_noisy * x, 'g--', label=f'МНК (SSE={sse_noisy:.2f})')
plt.plot(x, a_la_noisy + b_la_noisy * x, 'b--', label=f'МНМ (SAE={sae_noisy:.2f})')
plt.title('Возмущенная выборка')
plt.legend()

plt.tight_layout()
plt.show()