import numpy as np

from tools import padding


def max_p(input_matrix, pool_size=(2, 2), stride=None):
    i_shape = input_matrix.shape
    ph, pw = pool_size

    if stride is None:
        stride = max(ph, pw)

    h = (i_shape[0] - ph) // stride + 1
    w = (i_shape[1] - pw) // stride + 1

    pooled_matrix = np.zeros((h, w))

    for i in range(0, h):
        for j in range(0, w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + ph
            end_j = start_j + pw

            pooled_matrix[i, j] = np.max(input_matrix[start_i:end_i, start_j:end_j])

    return pooled_matrix


def cognitron_max_p_v1(input_matrix, pool_size=(2, 2)):
    # print(input_matrix.shape)

    pdh = int(pool_size[0]/2)
    pdw = int(pool_size[1]/2)
    __input_matrix = padding.padding(input_matrix, pdh, pdw)
    # print(__input_matrix.shape)
    np.set_printoptions(linewidth=200)

    # matrix_str = np.array2string(__input_matrix, max_line_width=200, formatter={'float_kind': lambda x: "%.2f" % x})

    # print(__input_matrix)
    # for i in range(len(__input_matrix)):
    #     print(__input_matrix[i])
    i_shape = __input_matrix.shape
    ph, pw = pool_size
    area_id = 0
    max_indexes = {}
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            current_pooling_area = __input_matrix[i:i + ph, j:j + pw]

            area_id += 1

            max_index = np.argwhere(current_pooling_area == np.max(current_pooling_area))
            if len(max_index) == 1:
                max_index = max_index[0]
                # print(max_index)
            else:
                max_indexes[(i, j)] = None
                continue

            # max_index = np.unravel_index(max_index_flat, current_pooling_area.shape)
            global_max_index = (
                max(0, i + max_index[0] - pdh),
                max(0, j + max_index[1] - pdw)
            )
            # print("max_index: ", max_index)
            # print("global_max_index: ", global_max_index)
            max_indexes[(i, j)] = global_max_index
    # print("area_id: ", area_id)
    # print("input_matrix: \n", input_matrix)
    # print("max_indexes", max_indexes)
    # print(len(max_indexes))
    return input_matrix, max_indexes


def cognitron_max_p_v2(input_matrix, pool_size=(2, 2)):
    i_shape = input_matrix.shape
    ph, pw = pool_size
    pooled_matrix = np.zeros((i_shape[0], i_shape[1]))

    max_indexes = []
    for i in range(0, i_shape[0], ph):
        for j in range(0, i_shape[1], pw):
            current_pooling_area = input_matrix[i:i + ph, j:j + pw]

            max_index_flat = np.argmax(current_pooling_area)
            max_index = np.unravel_index(max_index_flat, current_pooling_area.shape)
            global_max_index = (i + max_index[0], j + max_index[1])

            max_indexes.append(global_max_index)
    for i in range(i_shape[0]):
        for j in range(i_shape[1]):
            if (i, j) not in max_indexes:
                pooled_matrix[i, j] = 0
            else:
                pooled_matrix[i, j] = 1
    return pooled_matrix, max_indexes


def average_p(input_matrix, pool_size=(2, 2), stride=None):
    i_shape = input_matrix.shape
    ph, pw = pool_size

    if stride is None:
        stride = max(ph, pw)

    h = (i_shape[0] - ph) // stride + 1
    w = (i_shape[1] - pw) // stride + 1

    pooled_matrix = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + ph
            end_j = start_j + pw
            pooled_matrix[i, j] = np.mean(input_matrix[start_i:end_i, start_j:end_j])

    return pooled_matrix


def min_p(input_matrix, pool_size=(2, 2), stride=None):

    i_shape = input_matrix.shape
    ph, pw = pool_size

    if stride is None:
        stride = max(ph, pw)

    h = (i_shape[0] - ph) // stride + 1
    w = (i_shape[1] - pw) // stride + 1

    pooled_matrix = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + ph
            end_j = start_j + pw
            pooled_matrix[i, j] = np.min(input_matrix[start_i:end_i, start_j:end_j])

    return pooled_matrix
