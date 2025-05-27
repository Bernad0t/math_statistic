import numpy as np
from scipy.stats import norm, chi2, uniform
import pandas as pd
import matplotlib.pyplot as plt

# 1. Генерация выборки из равномерного распределения [a, b]
np.random.seed(42)
a, b = -3, 3  # границы равномерного распределения
uniform_sample = np.random.uniform(a, b, 100)

# 2. Оценка параметров для гипотетического нормального распределения
mu_hat = np.mean(uniform_sample)
sigma_hat = np.std(uniform_sample, ddof=1)

print(f"Оценки параметров для гипотетического нормального распределения:")
print(f"μ = {mu_hat:.4f}, σ = {sigma_hat:.4f}\n")

# 3. Проверка гипотезы о нормальности с помощью критерия χ²
bin_edges = [-np.inf, -1.1, -0.73, -0.37, 0.0, 0.37, 0.73, 1.1, np.inf]  # те же интервалы, что и в примере

observed, _ = np.histogram(uniform_sample, bins=bin_edges)
probs = np.diff(norm.cdf(bin_edges, mu_hat, sigma_hat))
expected = probs * len(uniform_sample)

# Создаем таблицу
table = pd.DataFrame({
    'i': range(1, len(bin_edges)),
    'Границы': [f"({bin_edges[i-1]:.2f}; {bin_edges[i]:.2f}]" for i in range(1, len(bin_edges))],
    'n_i (набл.)': observed,
    'p_i': probs.round(4),
    'np_i (ожид.)': expected.round(2),
    'n_i - np_i': (observed - expected).round(2),
    '(n_i - np_i)²/np_i': ((observed - expected)**2 / expected).round(4)
})

# Итоговая строка
total = pd.DataFrame({
    'i': ['∑'],
    'Границы': [''],
    'n_i (набл.)': [table['n_i (набл.)'].sum()],
    'p_i': [table['p_i'].sum().round(4)],
    'np_i (ожид.)': [table['np_i (ожид.)'].sum().round(2)],
    'n_i - np_i': [table['n_i - np_i'].sum().round(2)],
    '(n_i - np_i)²/np_i': [table['(n_i - np_i)²/np_i'].sum().round(4)]
})

final_table = pd.concat([table, total], ignore_index=True)

# Критерий χ²
chi2_stat = total['(n_i - np_i)²/np_i'].values[0]
critical_value = chi2.ppf(0.95, df=len(bin_edges)-3)  # df = k - 1 - 2 параметра (μ, σ)
p_value = 1 - chi2.cdf(chi2_stat, df=len(bin_edges)-3)

# Проверка гипотезы
alpha = 0.05
passed = chi2_stat < critical_value

# Вывод результатов
print("Таблица для критерия χ²:")
print(final_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nПроверка гипотезы о нормальности для равномерной выборки:")
print(f"χ² = {chi2_stat:.4f}")
print(f"Критическое значение (α=0.05) = {critical_value:.4f}")
print(f"p-value = {p_value:.6f}")

if passed:
    print("\nВывод: Гипотеза H₀ НЕ отвергается (данные выглядят нормально, но это ошибка!)")
    print("⚠️ Критерий χ² не смог обнаружить, что распределение равномерное!")
else:
    print("\nВывод: Гипотеза H₀ отвергается (данные НЕ нормальные, верный результат)")
    print("✅ Критерий корректно определил, что распределение равномерное")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(uniform_sample, bins=bin_edges, alpha=0.7, color='blue', edgecolor='black')
plt.title('Гистограмма равномерной выборки (n=100)')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
x = np.linspace(-4, 4, 1000)
plt.plot(x, norm.pdf(x, mu_hat, sigma_hat), 'r-', label=f'N({mu_hat:.2f}, {sigma_hat:.2f})')
plt.title('Гипотетическое нормальное распределение')
plt.legend()

plt.tight_layout()
plt.show()