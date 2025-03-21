import numpy as np

from tools import padding, pooling, activation_func as af
from typing import List


class CognitronLayer:

    def __init__(self, input_shape, receptive_area_shape, inhibiting_area_shape, competition_area_shape, layer_id):
        self.__id = layer_id

        self.__receptive_area_shape = receptive_area_shape
        self.__receptive_area_w_list = np.empty((input_shape[0], input_shape[1]), dtype=object)
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                self.__receptive_area_w_list[i, j] = np.zeros(receptive_area_shape)

        self.__inhibiting_area_shape = inhibiting_area_shape
        self.__inhibiting_area_w_list = np.empty((input_shape[0], input_shape[1]), dtype=object)
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                self.__inhibiting_area_w_list[i, j] = (np.random.dirichlet(np.ones(np.prod(inhibiting_area_shape)))
                                                       .reshape(inhibiting_area_shape))

        self.__inhibiting_impuls_w = np.zeros((input_shape[0], input_shape[1]))

        self.__competition_area_shape = competition_area_shape

    def summary(self):
        print(f"Cognitron layer №{self.__id}:")
        print(f"\tReceptive area shape: {self.__receptive_area_shape}")
        print(f"\tInhibiting area shape: {self.__inhibiting_area_shape}")
        print(f"\tCompetition area shape: {self.__competition_area_shape}")
        print(f"\tReceptive areas counts: {len(self.__receptive_area_w_list) ** 2}")

    def __forward(self, input_data):
        p_a_h = int(self.__receptive_area_shape[0] / 2)
        p_a_w = int(self.__receptive_area_shape[1] / 2)
        padded_for_activation_neurons = padding.padding(input_data, p_a_h, p_a_w)

        p_i_h = int(self.__inhibiting_area_shape[0] / 2)
        p_i_w = int(self.__inhibiting_area_shape[1] / 2)
        padded_for_inhibiting_neurons = padding.padding(input_data, p_i_h, p_i_w)

        return self.__proto_convolution(padded_for_activation_neurons, p_a_h, p_a_w,
                                        padded_for_inhibiting_neurons, p_i_h, p_i_w)

    def __proto_convolution(self, matrix_for_activating, p_a_h, p_a_w, matrix_for_inhibiting, p_i_h, p_i_w):
        output = []
        inhibiting_impuls_matrix = []
        activating_impuls_matrix = []
        for i in range(len(self.__receptive_area_w_list)):
            row = []
            i_row = []
            a_row = []
            for j in range(len(self.__receptive_area_w_list[i])):
                neuron_output, inhibiting_impuls, activating_impuls = self.__calculate_neuron_output(
                    matrix_for_activating[i: i + 2 * p_a_h + 1, j: j + 2 * p_a_w + 1],
                    self.__receptive_area_w_list[i, j],
                    matrix_for_inhibiting[i: i + 2 * p_i_h + 1, j: j + 2 * p_i_w + 1],
                    self.__inhibiting_area_w_list[i, j],
                    i, j
                )
                row.append(neuron_output)
                i_row.append(inhibiting_impuls)
                a_row.append(activating_impuls)
            output.append(row)
            inhibiting_impuls_matrix.append(i_row)
            activating_impuls_matrix.append(a_row)
        return (
            np.array(output, dtype=np.float64),
            np.array(inhibiting_impuls_matrix, dtype=np.float64),
            np.array(activating_impuls_matrix, dtype=np.float64)
        )

    def __calculate_neuron_output(self, receptive_area_data, receptive_area_w, inhibiting_area_data, inhibiting_area_w,
                                  i, j):
        activating_impuls = np.sum(np.dot(receptive_area_w, receptive_area_data))
        inhibiting_impuls = (np.sum(np.dot(inhibiting_area_w, inhibiting_area_data))
                             / (inhibiting_area_w.shape[0] * inhibiting_area_w.shape[1]))

        # max_v = np.finfo(np.float64).max / 10**20
        # min_v = np.finfo(np.float64).min / 10**20
        # activating_impuls = np.clip(np.sum(np.dot(receptive_area_w, receptive_area_data)), min_v, max_v)
        # inhibiting_impuls = np.clip(np.sum(np.dot(inhibiting_area_w, inhibiting_area_data)) / (
        #             inhibiting_area_w.shape[0] * inhibiting_area_w.shape[1]), min_v, max_v)

        weighed_inhibiting_impuls = self.__inhibiting_impuls_w[i, j] * inhibiting_impuls
        # print(activating_impuls, weighed_inhibiting_impuls)
        if weighed_inhibiting_impuls < 1:
            y = activating_impuls - weighed_inhibiting_impuls
        elif activating_impuls > 1 and weighed_inhibiting_impuls > 1:
            y = (activating_impuls / weighed_inhibiting_impuls) - 1
        else:
            y = (activating_impuls - weighed_inhibiting_impuls) / (1 + weighed_inhibiting_impuls)
        return af.relu(y), inhibiting_impuls, activating_impuls

    def __inhibit(self, matrix):
        return pooling.cognitron_max_p_v1(matrix, self.__competition_area_shape)

    def fit(self, input_data, q, q_prime):
        # print("input_data\n", input_data)
        output, inhibiting_impuls, activating_impuls = self.__forward(input_data)
        output, max_indexes = self.__inhibit(output)
        # print(max_indexes)
        self.__weights_change(input_data, max_indexes, inhibiting_impuls, activating_impuls, q, q_prime)
        # print("output\n", output)
        return output

    def predict(self, input_data):
        return self.__forward(input_data)[0]

    def __weights_change(self, input_data, indexes_to_change, inhibiting_output, activating_impuls, q, q_prime):
        # print(indexes_to_change)
        # print(inhibiting_output)

        for inhibiting_area_coordinates in indexes_to_change.keys():
            max_output_neuon_coordinates = indexes_to_change[inhibiting_area_coordinates]
            if max_output_neuon_coordinates is not None:
                # print("has activating neurons ", self.__id, inhibiting_area_coordinates, inhibiting_output[inhibiting_area_coordinates])
                center_i = inhibiting_area_coordinates[0]
                center_j = inhibiting_area_coordinates[1]
                i = max_output_neuon_coordinates[0] - self.get_coordinate_difference(center_i)
                j = max_output_neuon_coordinates[1] - self.get_coordinate_difference(center_j)
                # print(self.__receptive_area_w_list[inhibiting_area_coordinates][i, j])
                self.__receptive_area_w_list[inhibiting_area_coordinates][i, j] += (
                        q *
                        self.__inhibiting_impuls_w[i, j] *
                        input_data[i, j]
                )
                if inhibiting_output[inhibiting_area_coordinates] != 0:
                    self.__inhibiting_impuls_w[inhibiting_area_coordinates] = (
                            q *
                            activating_impuls[i, j] /
                            (2 * inhibiting_output[inhibiting_area_coordinates])
                    )
            else:
                # print("no activating neurons", self.__id)
                current_receptive_area = self.__receptive_area_w_list[inhibiting_area_coordinates]
                for i in range(len(current_receptive_area)):
                    for j in range(len(current_receptive_area[0])):
                        self.__receptive_area_w_list[inhibiting_area_coordinates][i, j] += (
                                q_prime *
                                self.__inhibiting_impuls_w[i, j] *
                                input_data[i, j]
                        )
                self.__inhibiting_impuls_w[inhibiting_area_coordinates] = (
                        q_prime *
                        inhibiting_output[inhibiting_area_coordinates]
                )

    def get_coordinate_difference(self, coordinate):
        receptive_area_radius = int(self.__receptive_area_shape[0] / 2)
        difference = coordinate - receptive_area_radius
        if difference == 0:
            return receptive_area_radius
        else:
            return difference


class CognitronNN:
    def __init__(self, layers: List[CognitronLayer]):
        self.__layers = layers

    def summary(self):
        for layer in self.__layers:
            layer.summary()

    def fit(self, train_data, q=16, q_prime=2, epochs=20):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for data in train_data:
                # print(data)
                output = data
                for layer in self.__layers:
                    output = layer.fit(output, q, q_prime)
                # print("-----------------------------------")

    def predict(self, input_data):
        predicted = np.array(input_data)
        for layer in self.__layers:
            predicted = layer.predict(input_data)
        max_index_flat = np.argmax(predicted)
        max_index_2d = np.unravel_index(max_index_flat, predicted.shape)
        result = np.zeros(predicted.shape)
        result[max_index_2d] = 1
        return result, max_index_2d

    def predict_v2(self, input_data):
        predicted = np.array(input_data)
        for layer in self.__layers:
            predicted = layer.predict(input_data)
        for layer in self.__layers[::-1]:
            predicted = layer.predict(predicted)
        return np.where(predicted != 0, 1, 0)

    def predict_v3(self, input_data):
        predicted = np.array(input_data)
        for layer in self.__layers:
            predicted = layer.predict(input_data)
        max_index_flat = np.argmax(predicted)
        max_index_2d = np.unravel_index(max_index_flat, predicted.shape)
        result = np.zeros(predicted.shape)
        result[max_index_2d] = 1
        for layer in self.__layers[::-1]:
            result = layer.predict(result)
        return np.where(result != 0, 1, 0)

    def evaluate(self):
        pass