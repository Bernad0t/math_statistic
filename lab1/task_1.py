import matplotlib.pyplot as plt
import seaborn as sns

from lab1.utils import generate_distributions


# Функция генерации распределений

class Task1:

    # Функция построения гистограммы
    def _plot_distribution(self, distribution, name, power, interval):
        plt.figure(figsize=(8, 6))
        sns.histplot(distribution[:power], bins=interval, kde=True, stat="density")
        #   bins=interval: Задает количество интервалов (корзин) для гистограммы.
        #   kde=True: Добавляет линию плотности (оценка ядровой плотности).
        #   stat="density": Масштабирует гистограмму так, чтобы сумма высот интервалов давала плотность (а не частоту).
        plt.title(f'{name}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
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
            "normal": [7, 20, 30],
            "cauchy": [5, 12, 15],
            "poisson": [3, 10, 15],
            "uniform": [3, 10, 15],
        }

        # Обработка каждого распределения
        for distribution, name in zip(distributions, names):
            for power, interval in zip(powers, intervals[name]):
                self._plot_distribution(distribution, name, power, interval)