import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr, spearmanr
from matplotlib.patches import Ellipse


def generate_samples_and_calculate(n_samples_list, rhos, n_iterations=1000):
    results = {}

    for n in n_samples_list:
        for rho in rhos:
            # Генерация для чистого нормального распределения
            pearson_vals = []
            spearman_vals = []
            quad_vals = []

            for _ in range(n_iterations):
                # Генерация данных
                cov = [[1, rho], [rho, 1]]
                data = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=n)

                # Вычисление коэффициентов
                pearson_r, _ = pearsonr(data[:, 0], data[:, 1])
                spearman_r, _ = spearmanr(data[:, 0], data[:, 1])
                quad_r = np.corrcoef(data[:, 0] ** 2, data[:, 1] ** 2)[0, 1]

                pearson_vals.append(pearson_r)
                spearman_vals.append(spearman_r)
                quad_vals.append(quad_r)

            # Сохранение результатов
            key = f"normal_n{n}_rho{rho}"
            results[key] = {
                'pearson_mean': np.mean(pearson_vals),
                'pearson_var': np.var(pearson_vals),
                'spearman_mean': np.mean(spearman_vals),
                'spearman_var': np.var(spearman_vals),
                'quad_mean': np.mean(quad_vals),
                'quad_var': np.var(quad_vals),
                'data': data
            }

    return results


def generate_mixture_samples(n_samples_list, n_iterations=1000):
    results = {}

    for n in n_samples_list:
        pearson_vals = []
        spearman_vals = []
        quad_vals = []

        for _ in range(n_iterations):
            # Генерация смеси распределений
            data1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.9], [0.9, 1]], size=int(0.9 * n))
            data2 = np.random.multivariate_normal(mean=[0, 0], cov=[[10, -0.9], [-0.9, 10]], size=int(0.1 * n))
            data = np.vstack((data1, data2))

            # Вычисление коэффициентов
            pearson_r, _ = pearsonr(data[:, 0], data[:, 1])
            spearman_r, _ = spearmanr(data[:, 0], data[:, 1])
            quad_r = np.corrcoef(data[:, 0] ** 2, data[:, 1] ** 2)[0, 1]

            pearson_vals.append(pearson_r)
            spearman_vals.append(spearman_r)
            quad_vals.append(quad_r)

        # Сохранение результатов
        key = f"mixture_n{n}"
        results[key] = {
            'pearson_mean': np.mean(pearson_vals),
            'pearson_var': np.var(pearson_vals),
            'spearman_mean': np.mean(spearman_vals),
            'spearman_var': np.var(spearman_vals),
            'quad_mean': np.mean(quad_vals),
            'quad_var': np.var(quad_vals),
            'data': data
        }

    return results


def plot_contours(data, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)

    # Отрисовка эллипсов равной плотности
    cov = np.cov(data.T)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)

    for j in range(1, 4):
        ell = Ellipse(xy=(np.mean(data[:, 0]), np.mean(data[:, 1])),
                      width=lambda_[0] * j * 2, height=lambda_[1] * j * 2,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      edgecolor='red', fc='None', lw=2)
        plt.gca().add_patch(ell)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


# Параметры исследования
n_samples_list = [20, 60, 100]
rhos = [0, 0.5, 0.9]

# Выполнение расчетов
normal_results = generate_samples_and_calculate(n_samples_list, rhos)
mixture_results = generate_mixture_samples(n_samples_list)

# Визуализация результатов
for key in normal_results:
    print(f"\nРезультаты для {key}:")
    print(f"Пирсон: mean={normal_results[key]['pearson_mean']:.4f}, var={normal_results[key]['pearson_var']:.4f}")
    print(f"Спирмен: mean={normal_results[key]['spearman_mean']:.4f}, var={normal_results[key]['spearman_var']:.4f}")
    print(f"Квадратичный: mean={normal_results[key]['quad_mean']:.4f}, var={normal_results[key]['quad_var']:.4f}")
    plot_contours(normal_results[key]['data'], key)

for key in mixture_results:
    print(f"\nРезультаты для {key}:")
    print(f"Пирсон: mean={mixture_results[key]['pearson_mean']:.4f}, var={mixture_results[key]['pearson_var']:.4f}")
    print(f"Спирмен: mean={mixture_results[key]['spearman_mean']:.4f}, var={mixture_results[key]['spearman_var']:.4f}")
    print(f"Квадратичный: mean={mixture_results[key]['quad_mean']:.4f}, var={mixture_results[key]['quad_var']:.4f}")
    plot_contours(mixture_results[key]['data'], key)