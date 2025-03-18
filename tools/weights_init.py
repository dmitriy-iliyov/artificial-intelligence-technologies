import numpy as np


def default_init(shape):
    return np.random.uniform(-1, 1, size=shape)


def xavier_init(shape):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, shape)


def he_init(shape):
    stddev = np.sqrt(2 / shape[0])
    return np.random.randn(*shape) * stddev
