import numpy as np

from tools import padding, pooling, activation_func as af
from typing import List


class ConvolutionLayer:

    def __init__(self, filters_count, specters_count, filter_shape, weights_init_func, activation_func, id):
        self.__id = id
        self.__filters_w_list = np.empty((filters_count, specters_count), dtype=object)
        for filter_id in range(filters_count):
            for filter_part_id in range(specters_count):
                self.__filters_w_list[filter_id, filter_part_id] = weights_init_func(filter_shape)
        self.__filters_b_list = np.zeros((filters_count,))
        self.__activation_function = activation_func

    def summary(self):
        print(f"Convolution Layer №{self.__id}")
        print(f"Number of filters: {self.__filters_w_list.shape[0]}")
        print(f"Channels per filter: {self.__filters_w_list.shape[1]}")
        print(f"Filter shape: {self.__filters_w_list[0, 0].shape}")
        print(f"Activation function: {self.__activation_function.__name__}")
        print(f"Bias shape: {self.__filters_b_list.shape}")
        print(f"Bias values (first 5): {self.__filters_b_list[:5]}")

    def summary_for_test(self):
        print(f"Bias values (first 5): {self.__filters_b_list[:5]}")