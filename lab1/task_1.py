import matplotlib.pyplot as plt
import seaborn as sns
import math
from lab1.utils import generate_distributions


# Функция генерации распределений

class Task1:

    # Функция построения гистограммы
    def _plot_distribution(self, distribution, name, power, interval):
        plt.figure(figsize=(8, 6))
        sns.histplot(distribution[:power], bins=interval, kde=name.lower() != "uniform", stat="density")

        # Условие для равномерного распределения
        if name.lower() == "uniform":
            a, b = -math.sqrt(3), math.sqrt(3)  # Теоретические границы распределения
            density = 1 / (b - a)  # Плотность равномерного распределения
            # Добавляем единственную горизонтальную линию на интервале [a, b]
            plt.plot([a, b], [density, density], color='red', linestyle='--', label=f'PDF {density:.3f}')

        plt.title(f'{name}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        if name.lower() == "uniform":
            plt.legend()
        plt.savefig(f'images_task_1/{name}_size_{power}.png')
        plt.close()

    # Основная функция для обработки всех распределений
    def process_distributions(self):
        # Генерация распределений
        distributions = generate_distributions()
        names = ["normal", "cauchy", "poisson", "uniform"]

        # Определения параметров: размеры выборок, интервалы гистограмм.
        powers = [10, 50, 1000]
        intervals = {
            "normal": [4, 10, 30],
            "cauchy": [2, 5, 30],
            "poisson": [3, 10, 15],
            "uniform": [3, 10, 15],
        }

        # Обработка каждого распределения
        for distribution, name in zip(distributions, names):
            for power, interval in zip(powers, intervals[name]):
                self._plot_distribution(distribution, name, power, interval)