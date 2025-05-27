import numpy as np
from scipy.stats import norm, t, chi2
from scipy.stats import bootstrap


def calculate_confidence_intervals(sample, alpha=0.05):
    n = len(sample)
    x_bar = np.mean(sample)
    s = np.std(sample, ddof=1)

    # Для нормального распределения
    if n <= 30:
        # t-распределение для малых выборок
        t_val = t.ppf(1 - alpha / 2, n - 1)
        ci_m_normal = (x_bar - t_val * s / np.sqrt(n - 1),
                       x_bar + t_val * s / np.sqrt(n - 1))

        # Хи-квадрат для дисперсии
        chi2_low = chi2.ppf(1 - alpha / 2, n - 1)
        chi2_high = chi2.ppf(alpha / 2, n - 1)
        ci_sigma_normal = (s * np.sqrt(n) / np.sqrt(chi2_low),
                           s * np.sqrt(n) / np.sqrt(chi2_high))
    else:
        # Нормальное распределение для больших выборок
        z_val = norm.ppf(1 - alpha / 2)
        ci_m_normal = (x_bar - z_val * s / np.sqrt(n),
                       x_bar + z_val * s / np.sqrt(n))

        # Асимптотический подход для дисперсии
        e = (np.mean((sample - x_bar) ** 4) / s ** 4) - 3  # эксцесс
        U = z_val * np.sqrt((e + 2) / n)
        ci_sigma_normal = (s * (1 + U) ** (-0.5), s * (1 - U) ** (-0.5))

    # Для произвольного распределения (асимптотический подход)
    z_val = norm.ppf(1 - alpha / 2)
    ci_m_asymptotic = (x_bar - z_val * s / np.sqrt(n),
                       x_bar + z_val * s / np.sqrt(n))

    # Для дисперсии произвольного распределения
    if n > 30:
        e = (np.mean((sample - x_bar) ** 4) / s ** 4) - 3
        U = z_val * np.sqrt((e + 2) / n)
        ci_sigma_asymptotic = (s * (1 - 0.5 * U), s * (1 + 0.5 * U))
    else:
        # Бутстреп для малых выборок
        def std_statistic(data):
            return np.std(data, ddof=1)

        bootstrap_ci = bootstrap((sample,), std_statistic,
                                 confidence_level=1 - alpha,
                                 n_resamples=9999,
                                 method='percentile')
        ci_sigma_asymptotic = (bootstrap_ci.confidence_interval.low,
                               bootstrap_ci.confidence_interval.high)

    return {
        'normal': {'mean': ci_m_normal, 'std': ci_sigma_normal},
        'asymptotic': {'mean': ci_m_asymptotic, 'std': ci_sigma_asymptotic}
    }


# Генерация выборок
np.random.seed(42)
sample_20 = np.random.normal(0, 1, 20)
sample_100 = np.random.normal(0, 1, 100)

# Расчет доверительных интервалов
ci_20 = calculate_confidence_intervals(sample_20)
ci_100 = calculate_confidence_intervals(sample_100)


# Форматирование результатов
def format_interval(interval):
    return f"{interval[0]:.2f} < m < {interval[1]:.2f}" if 'mean' in interval else f"{interval[0]:.2f} < σ < {interval[1]:.2f}"


# Вывод результатов в виде таблиц
print("Таблица 1: Доверительные интервалы для параметров нормального распределения")
print(f"| n = 20  | {format_interval(ci_20['normal']['mean'])} | {format_interval(ci_20['normal']['std'])} |")
print(f"| n = 100 | {format_interval(ci_100['normal']['mean'])} | {format_interval(ci_100['normal']['std'])} |")

print("\nТаблица 2: Доверительные интервалы для параметров произвольного распределения")
print(f"| n = 20  | {format_interval(ci_20['asymptotic']['mean'])} | {format_interval(ci_20['asymptotic']['std'])} |")
print(f"| n = 100 | {format_interval(ci_100['asymptotic']['mean'])} | {format_interval(ci_100['asymptotic']['std'])} |")