import numpy as np


def padding(input_matrix, p_h, p_w):
    input_h, input_w = input_matrix.shape
    padded_h, padded_w = input_h + 2 * p_h, input_w + 2 * p_w
    new_matrix = np.zeros((padded_h, padded_w), dtype=np.float64)
    new_matrix[p_h:p_h + input_h, p_w:p_w + input_w] = input_matrix
    return new_matrix
