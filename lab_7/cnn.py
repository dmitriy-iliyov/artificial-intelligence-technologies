from abc import ABC

import numpy as np

from lab_2.perceptrons import NotWorkMultiCategoricalMLP
from tools import padding, pooling
from tools import weights_init, activation_func
from lab_2.perceptrons import Layer


class ConvolutionLayer:

    def __init__(self, filters_count, kernel_shape, stride, n_activation_func, channels_count,
                 weights_init_func=weights_init.default_init):
        self.__kernel_shape = kernel_shape
        self.__filters_w_list = []
        self.__filters_w_list = [weights_init_func((*kernel_shape, channels_count)) for _ in range(filters_count)]
        self.__filters_b_list = np.zeros((filters_count,))
        self.__stride = stride
        self.__activation_function = n_activation_func

    def summary(self):
        print(f"Convolution Layer")
        print(f"\tNumber of filters: {len(self.__filters_w_list)}")
        print(f"\tChannels per filter: {len(self.__filters_w_list[0])}")
        print(f"\tKernel shape: {self.__filters_w_list[0][0].shape}")
        print(f"\tActivation function: {self.__activation_function.__name__}")
        print(f"\tBias shape: {self.__filters_b_list.shape}")
        print(f"\tBias values (first 5): {self.__filters_b_list[:5]}")

    def summary_for_test(self):
        print(f"Bias values (first 5): {self.__filters_b_list[:5]}")

    # def forward(self, input_channels):
    #     output_channels = []
    #     for i in range(len(self.__filters_w_list)):
    #         channel = self.__convolution(input_channels, self.__filters_w_list[i])
    #         print(f"channel {channel}")
    #         # channel = np.squeeze(channel, axis=-1)  # Это удалит ось канала, если она равна 1
    #         # print(f"channel {channel}")
    #         channel += self.__filters_b_list[i]
    #         channel = self.__activation_function(channel)
    #         output_channels.append(channel)
    #         output_channels.append(channel)
    #     print(output_channels)
    #     output_channels = np.stack(output_channels, axis=-1)
    #     print(output_channels)
    #     return output_channels
    #
    # def __convolution(self, input_channels, _filter):
    #     result_channels = []
    #     for ci in range(input_channels.shape[2]):
    #         result_matrix = []
    #         current_matrix = input_channels[:, :, ci]
    #         print(current_matrix)
    #         for row_i in range(0, current_matrix.shape[0] - self.__kernel_shape[0] + 1, self.__stride):
    #             row = []
    #             for col_i in range(0, current_matrix.shape[1] - self.__kernel_shape[1] + 1, self.__stride):
    #                 conv_result = np.sum(
    #                     current_matrix[row_i: row_i + self.__kernel_shape[0], col_i: col_i + self.__kernel_shape[1]] *
    #                     _filter[ci]
    #                 )
    #                 row.append(conv_result)
    #             result_matrix.append(row)
    #         result_channels.append(result_matrix)
    #     result_channels = np.array(result_channels, dtype=np.float64)
    #     print(f"result_channels: {result_channels}")
    #     return np.sum(result_channels, axis=0)

    def forward(self, input_channels):
        print("ConvolutionLayer")
        print(f"\tinput_channels.shape={input_channels.shape}")
        # print(input_channels)
        output_channels = []
        for i in range(len(self.__filters_w_list)):
            channel = self.__convolution(input_channels, self.__filters_w_list[i])
            channel += self.__filters_b_list[i]
            channel = self.__activation_function(channel)
            output_channels.append(channel)
        output_channels = np.stack(output_channels, axis=-1)
        print(f"\toutput_channels.shape={output_channels.shape}")
        return output_channels

    def __convolution(self, input_channels, _filter):
        result_matrix = []
        for ri in range(0, input_channels.shape[0] - self.__kernel_shape[0] + 1, self.__stride):
            row = []
            for ci in range(0, input_channels.shape[1] - self.__kernel_shape[1] + 1, self.__stride):
                region = input_channels[ri: ri + self.__kernel_shape[0], ci: ci + self.__kernel_shape[1], :]
                conv_result = np.sum(region * _filter)
                row.append(conv_result)
            result_matrix.append(row)
        return np.array(result_matrix, dtype=np.float64)

    def backward(self):
        pass


class PaddingLayer:

    def __init__(self, shape):
        self.__shape = shape

    def forward(self, input_channels):
        print("PaddingLayer")
        print(f"\tinput_channels.shape={input_channels.shape}")
        # print(input_channels)
        if len(input_channels.shape) == 2:
            return padding.padding(input_channels, self.__shape[0], self.__shape[1])
        else:
            padded_channels = [padding.padding(input_channels[:, :, c], self.__shape[0], self.__shape[1])
                               for c in range(input_channels.shape[2])]
            output_channels = np.stack(padded_channels, axis=-1)
            print(f"\toutput_channels.shape={output_channels.shape}")
            return output_channels

    def backward(self):
        pass


class MaxPoolingLayer:
    def __init__(self, shape, stride):
        self.__shape = shape
        self.__stride = stride

    def forward(self, input_channels):
        print("MaxPoolingLayer")
        print(f"\tinput_channels.shape={input_channels.shape}")
        # print(input_channels)
        if len(input_channels.shape) == 2:
            return pooling.max_p(input_channels, self.__shape, self.__stride)
        else:
            pooled_channels = [pooling.max_p(input_channels[:, :, c], self.__shape, self.__stride)
                               for c in range(input_channels.shape[2])]
            output_channels = np.stack(pooled_channels, axis=-1)
            print(f"\toutput_channels.shape={output_channels.shape}")
            return output_channels

    def backward(self):
        pass


class AveragePoolingLayer:
    def __init__(self, shape, stride):
        self.__shape = shape
        self.__stride = stride

    def forward(self, input_channels):
        print("AveragePoolingLayer")
        print(f"\tinput_channels.shape={input_channels.shape}")
        # print(input_channels)
        if len(input_channels.shape) == 2:
            return pooling.average_p(input_channels, self.__shape, self.__stride)
        else:
            pooled_channels = [pooling.average_p(input_channels[:, :, c], self.__shape, self.__stride)
                               for c in range(input_channels.shape[2])]
            output_channels = np.stack(pooled_channels, axis=-1)
            print(f"\toutput_channels.shape={output_channels.shape}")
            return output_channels

    def backward(self):
        pass


class FlattenLayer(Layer, ABC):

    def __init__(self):
        self.__input_shape = None

    def summary(self):
        print("\nFlatten Layer")

    def forward(self, input_channels):
        self.__input_shape = input_channels.shape
        return input_channels.reshape(input_channels.shape[0], -1)

    def backward(self, input_gradient, alpha):
        return input_gradient.reshape(self.__input_shape)


class LeNet:

    def __init__(self):
        self.__layers = [
            PaddingLayer((2, 2)),
            ConvolutionLayer(6, (5, 5), 1, activation_func.sigmoid,
                             channels_count=1, weights_init_func=weights_init.default_init),
            AveragePoolingLayer((2, 2), 2),
            ConvolutionLayer(16, (5, 5), 1, activation_func.sigmoid,
                             channels_count=6, weights_init_func=weights_init.default_init),
            AveragePoolingLayer((2, 2), 2),
            FlattenLayer(),
            NotWorkMultiCategoricalMLP(400, [120, 84], activation_func.sigmoid,
                                       10, activation_func.softmax)
        ]

    def fit(self, train_data, train_answ, batch_size, epochs, learning_rate):
        for epoch in range(epochs):

            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answ = train_answ[indexes]

            for i in range(0, len(train_data) - batch_size, batch_size):
                batch_data = train_data[i: i + batch_size]
                batch_answ = train_answ[i: i + batch_size]
                batch_output = None
                for layer in self.__layers:
                    batch_output = layer.forward(batch_output)
                batch_error = batch_output

                for layer in self.__layers[:-1]:
                    batch_error = layer.backward()

            pass

    def predict(self, input_channels):
        for layer in self.__layers:
            input_channels = layer.forward(input_channels)
        return input_channels
