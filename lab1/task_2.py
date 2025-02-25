import numpy as np
import pandas as pd

from lab1.utils import generate_distributions


class Task2:
    # Функция для расчета квартильного среднего z_Q
    def _quartile_mean(self, data):
        """
        Рассчитывает квартильное среднее z_Q.
        """
        z1_4 = np.percentile(data, 25)  # Первый квартиль (25-й перцентиль)
        z3_4 = np.percentile(data, 75)  # Третий квартиль (75-й перцентиль)
        return (z1_4 + z3_4) / 2


    # Основная функция обработки данных
    def process_task_2(self):
        distributions = generate_distributions()
        names = ["normal", "cauchy", "poisson", "uniform"]
        sample_sizes = [10, 100, 1000]  # Заданные размеры выборок
        repetitions = 1000  # Число повторений

        # Таблица для накопления результатов
        results = []

        for dist, name in zip(distributions, names):  # Перебираем все распределения
            for size in sample_sizes:  # Перебираем размеры выборок
                # Списки для сбора статистик
                means = []  # Средние
                medians = []  # Медианы
                z_qs = []  # z_Q

                # Готовим 1000 выборок
                for _ in range(repetitions):
                    sample = np.random.choice(dist, size=size, replace=False)  # Выборка случайных значений
                    means.append(np.mean(sample))  # Считаем среднее
                    medians.append(np.median(sample))  # Считаем медиану
                    z_qs.append(self._quartile_mean(sample))  # Рассчитываем z_Q

                # Вычисляем итоговые параметры
                math_expect = np.mean(means)  # Математическое ожидание E(z)
                variance = np.mean(np.square(means)) - np.square(math_expect)  # Оценка дисперсии D(z)
                mean_medians = np.mean(medians)  # Средняя медиана
                mean_z_qs = np.mean(z_qs)  # Средний z_Q

                # Сохраняем результаты в таблицу
                results.append({
                    "Distribution": name,
                    "Sample Size": size,
                    "E(z)": math_expect,
                    "D(z)": variance,
                    "Mean of Medians": mean_medians,
                    "Mean of z_Q": mean_z_qs
                })

        # Возвращаем результаты как DataFrame
        pd.set_option('display.max_rows', None)  # Показываем все строки
        pd.set_option('display.max_columns', None)  # Показываем все столбцы
        pd.set_option('display.width', 1000)  # Увеличиваем ширину вывода
        pd.set_option('display.colheader_justify', 'center')  # Центрируем заголовки
        pd.set_option('display.float_format', '{:.4f}'.format)  # Формат чисел с 4 знаками после запятой
        return pd.DataFrame(results)

