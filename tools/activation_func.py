import numpy as np


def __clean_input(x):
    return np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)


def activation_threshold(x):
    return np.where(np.asarray(x) >= 0, 1, 0)


def activation_relu(x, df=False):
    x = np.asarray(x)
    if df:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)


def activation_leaky_relu(x, alpha=0.01, df=False):
    x = np.asarray(x)
    x = __clean_input(x)
    if df:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)


def activation_elu(x, alpha=1.0, df=False):
    x = np.asarray(x)
    x = __clean_input(x)
    if df:
        return np.where(x > 0, 1, activation_elu(x, alpha) + alpha)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def activation_prelu(x, alpha=0.01, df=False):
    x = np.asarray(x)
    x = __clean_input(x)
    if df:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)


def activation_tanh(x, df=False):
    x = np.asarray(x)
    tanh_x = np.tanh(x)
    if df:
        return 1 - tanh_x ** 2
    return tanh_x


def activation_sigmoid(x, df=False):
    x = np.asarray(x)
    sigmoid = 1 / (1 + np.exp(-x))
    if df:
        return sigmoid * (1 - sigmoid)
    return sigmoid


def softmax(x, df=False):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_output = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    if df:
        s = softmax_output
        jacobian_matrix = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[1]):
                    if j == k:
                        jacobian_matrix[i, j, k] = s[i, j] * (1 - s[i, j])
                    else:
                        jacobian_matrix[i, j, k] = -s[i, j] * s[i, k]
        return jacobian_matrix

    return softmax_output
