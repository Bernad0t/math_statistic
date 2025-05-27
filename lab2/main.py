import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Определяем параметры распределений
distributions = {
    "Normal": lambda size: np.random.normal(0, 1, size),
    "Cauchy": lambda size: np.random.standard_cauchy(size),
    "Poisson": lambda size: np.random.poisson(10, size),
    "Uniform": lambda size: np.random.uniform(-np.sqrt(3), np.sqrt(3), size),
}

# Размеры выборок
sample_sizes = [20, 100, 1000]

# Подготовка для хранения количества выбросов
outliers_data = []

# Генерация графиков
fig, axes = plt.subplots(len(sample_sizes), len(distributions), figsize=(15, 10), constrained_layout=True)

for i, n in enumerate(sample_sizes):
    for j, (name, dist) in enumerate(distributions.items()):
        # Генерируем выборку
        sample = dist(n)

        # Построение боксплота для данной выборки
        ax = axes[i, j]
        ax.boxplot(sample, vert=False)
        ax.set_title(f"{name} (n={n})")

        # Вычисляем выбросы
        q1, q3 = np.percentile(sample, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = sample[(sample < lower_bound) | (sample > upper_bound)]

        # Сохраняем данные о выбросах
        outliers_data.append({
            "Distribution": name,
            "Sample Size": n,
            "Outliers": len(outliers),
            "Outliers Percentage (%)": len(outliers) / n * 100
        })

# Показ графиков
plt.suptitle("", fontsize=16, y=1.02)
plt.show()

# Таблица с данными о выбросах
df_outliers = pd.DataFrame(outliers_data)
print(df_outliers)

# Анализ данных
print("\nСводная таблица выбросов")
print(df_outliers.pivot_table(index="Distribution", columns="Sample Size",
                              values=["Outliers", "Outliers Percentage (%)"]))
