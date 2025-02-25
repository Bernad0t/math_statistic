import numpy as np
import math


def generate_distributions(seed=30):
    np.random.seed(seed)
    normal = np.random.standard_normal(1000)
    cauchy = np.random.standard_cauchy(1000)
    poisson = np.random.poisson(10, size=1000)
    uniform = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 1000)
    return normal, cauchy, poisson, uniform
