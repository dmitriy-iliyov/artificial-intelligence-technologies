from abc import ABC, abstractmethod
from typing import List

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tools import weights_init as wi, activation_func as af


# matplotlib.use('TkAgg')


class Layer(ABC):

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def forward(self, input_vector):
        pass

    @abstractmethod
    def backward(self, input_errors, alpha):
        pass


class NonLearnableSLP:
    def __init__(self, weights, bias, activation_func):
        self.__activation_func = activation_func
        self.__weights = weights
        self.__b = bias

    def involve(self, inputs):
        return self.__activation_func(np.dot(inputs, self.__weights) + self.__b)

    def summary(self):
        print("weights =", self.__weights)
        print("bias =", self.__b)


class LearnableSLP:

    def __init__(self, activation_func):
        self.__activation_func = activation_func
        self.__weights = wi.default_init((2,))
        self.__b = 0
        self.__accuracy = None
        self.__loss_history = []

    def summary(self):
        print("weights =", self.__weights)
        print("bias =", self.__b)

    def __fit_forward(self, input_data):
        input_sum = np.dot(input_data, self.__weights)
        return self.__activation_func(input_sum + self.__b), input_sum

    def fit(self, train_data, train_answ, epochs=50, accuracy=0.1, learning_rate=0.01):
        self.__accuracy = accuracy
        for epoch in range(1, epochs + 1):
            total_error = 0
            for _ci, case in enumerate(train_data):
                prediction, input_sum = self.__fit_forward(case)
                error = train_answ[_ci] - prediction
                delta = error * self.__activation_func(input_sum, True)
                self.__weights -= learning_rate * delta * case
                self.__b -= learning_rate * delta
                total_error += abs(error)

            avg_error = total_error / len(train_data)
            self.__loss_history.append(avg_error)
        self.plot_loss()

    def predict(self, input_data):
        return af.threshold(np.dot(input_data, self.__weights) + self.__b)

    def evaluate(self, test_data, test_answ):
        correct = 0
        for x, y in zip(test_data, test_answ):
            if abs(self.predict(x) - y) <= self.__accuracy:
                correct += 1
        return correct / len(test_data)

    def plot_loss(self):
        plt.plot(self.__loss_history, label='Training Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error changing')
        plt.legend()
        plt.show()


class BinaryMLP:

    def __init__(self, input_n_count, hidden_layers, hidden_n_activation_func, output_n_activation_func):

        if len(hidden_layers) == 0:
            raise TypeError("hidden_layers shouldn't be empty")
        if not isinstance(hidden_layers, list):
            print("\033[93mWARNING:\033[0m hidden_layers should be list, but his type is: ", type(hidden_layers))
            hidden_layers = list(hidden_layers)
            print("converted to list: ", hidden_layers)

        self.__input_n_count = input_n_count

        self.__hidden_w_list = []
        self.__hidden_b_list = []
        previous_n_count = input_n_count
        for n_count in hidden_layers:
            self.__hidden_w_list.append(wi.default_init((previous_n_count, n_count)))
            self.__hidden_b_list.append(np.zeros((n_count,)))
            previous_n_count = n_count

        self.__output_w_list = wi.default_init((previous_n_count,))
        self.__output_b_list = np.zeros((1,))

        self.__hidden_n_activation_func = hidden_n_activation_func
        self.__output_n_activation_func = output_n_activation_func

        self.__learning_rate = None
        self.__loss_history = []

    def summary(self, full=False):
        print("Input neurons count:", self.__input_n_count)
        print("Hidden layers count:", len(self.__hidden_w_list))

        for i, (w, b) in enumerate(zip(self.__hidden_w_list, self.__hidden_b_list)):
            print(f"\nHidden Layer {i + 1}:")
            print(f"  Weights shape: {w.shape}")
            print(f"  Bias shape: {b.shape}")
            if full:
                print(f"  Weights:\n{w}")
                print(f"  Bias:\n{b}")

        print("\nOutput Layer:")
        print(f"  Weights shape: {self.__output_w_list.shape}")
        print(f"  Bias shape: {self.__output_b_list.shape}")
        if full:
            print(f"  Weights:\n{self.__output_w_list}")
            print(f"  Bias:\n{self.__output_b_list}")

    def __fit_forward(self, batch):
        batch_outputs = []
        batch_extra_output = []

        for _vec in batch:
            output = _vec
            current_layer_outputs = [output]

            for i, weights in enumerate(self.__hidden_w_list):
                subtotal = np.dot(output, weights) + self.__hidden_b_list[i]
                output = self.__hidden_n_activation_func(subtotal)
                current_layer_outputs.append(output)

            output_subtotal = np.dot(output, self.__output_w_list) + self.__output_b_list
            batch_outputs.append(self.__output_n_activation_func(output_subtotal))
            batch_extra_output.append(current_layer_outputs)

        return np.asarray(batch_outputs, dtype=np.float64), batch_extra_output

    def fit(self, train_data, train_answ, batch_size, epochs, learning_rate):

        self.__learning_rate = learning_rate

        for epoch in range(1, epochs + 1):

            epoch_error = []

            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answ = train_answ[indexes]

            for i in range(0, len(train_data), batch_size):
                batch_outputs, batch_extra_output = self.__fit_forward(train_data[i:i + batch_size])
                error = self.__back_propagation(batch_outputs,
                                                batch_extra_output,
                                                train_answ[i:i + batch_size])
                if isinstance(error, list):
                    epoch_error.extend(abs(e) for e in error)
                else:
                    epoch_error.append(abs(error))
            avg_error = np.mean(epoch_error)
            self.__loss_history.append(avg_error)

    def __back_propagation(self, batch_outputs, batch_extra_output, batch_answ):

        batch_answ = np.array(batch_answ, dtype=np.float64).reshape(-1, 1)
        errors = batch_outputs - batch_answ
        error = np.mean(errors)

        deltas = error * self.__output_n_activation_func(batch_outputs, True)
        delta = np.mean(deltas)

        delta_previous_layer_output = np.array([row[-1] for row in batch_extra_output], dtype=np.float64).mean(axis=0)
        self.__output_w_list -= self.__learning_rate * delta * delta_previous_layer_output
        self.__output_b_list -= self.__learning_rate * delta
        next_deltas = (delta * self.__output_w_list *
                       self.__hidden_n_activation_func(delta_previous_layer_output, True))
        next_delta = np.mean(next_deltas)

        for i in range(len(self.__hidden_w_list) - 1, -1, -1):
            delta_previous_layer_output = np.array([row[i] for row in batch_extra_output], dtype=np.float64).mean(
                axis=0)
            delta_previous_layer_output = delta_previous_layer_output.reshape(-1, 1)
            self.__hidden_w_list[i] -= self.__learning_rate * delta_previous_layer_output * next_delta
            self.__hidden_b_list[i] -= self.__learning_rate * next_delta
            next_deltas = (next_delta * self.__hidden_w_list[i] *
                           self.__hidden_n_activation_func(delta_previous_layer_output, True))
            next_delta = np.mean(next_deltas)

        return errors

    def predict(self, input_data):
        output = input_data
        for i, weights in enumerate(self.__hidden_w_list):
            output = self.__hidden_n_activation_func(np.dot(output, weights) + self.__hidden_b_list[i])
        output = self.__output_n_activation_func(np.dot(output, self.__output_w_list) + self.__output_b_list)
        return round(output[0])

    def evaluate(self, test_data, test_answ):
        correct = 0
        for x, y in zip(test_data, test_answ):
            if self.predict(x) == y:
                correct += 1
        return correct / len(test_data)

    def plot_loss(self):
        plt.plot(self.__loss_history, label='Training Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error changing')
        plt.legend()
        plt.show()


class NotWorkMultiCategoricalMLP:

    def __init__(self, input_n_count, hidden_layers, hidden_n_activation_func, output_n_count,
                 output_n_activation_func):

        if len(hidden_layers) == 0:
            raise TypeError("hidden_layers shouldn't be empty")
        if not isinstance(hidden_layers, list):
            print("\033[93mWARNING:\033[0m hidden_layers should be list, but his type is: ", type(hidden_layers))
            hidden_layers = list(hidden_layers)
            print("converted to list: ", hidden_layers)

        self.__input_n_count = input_n_count

        self.__hidden_w_list = []
        self.__hidden_b_list = []
        previous_n_count = input_n_count
        for n_count in hidden_layers:
            self.__hidden_w_list.append(wi.default_init((previous_n_count, n_count)))
            self.__hidden_b_list.append(np.zeros((n_count,)))
            previous_n_count = n_count

        self.__output_w_list = wi.default_init((previous_n_count, output_n_count))
        self.__output_b_list = np.zeros((output_n_count,))

        self.__hidden_n_activation_func = hidden_n_activation_func
        self.__output_n_activation_func = output_n_activation_func

        self.__learning_rate = None
        self.__loss_history = []

    def summary(self, full=False):
        print("Input neurons count:", self.__input_n_count)
        print("Hidden layers count:", len(self.__hidden_w_list))

        for i, (w, b) in enumerate(zip(self.__hidden_w_list, self.__hidden_b_list)):
            print(f"\nHidden Layer {i + 1}:")
            print(f"  Weights shape: {w.shape}")
            print(f"  Bias shape: {b.shape}")
            if full:
                print(f"  Weights:\n{w}")
                print(f"  Bias:\n{b}")

        print("\nOutput Layer:")
        print(f"  Weights shape: {self.__output_w_list.shape}")
        print(f"  Bias shape: {self.__output_b_list.shape}")
        if full:
            print(f"  Weights:\n{self.__output_w_list}")
            print(f"  Bias:\n{self.__output_b_list}")

    def forward(self, batch):
        batch_outputs = []

        for _vec in batch:
            output = _vec
            current_layer_outputs = [output]

            for i, weights in enumerate(self.__hidden_w_list):
                subtotal = np.dot(output, weights) + self.__hidden_b_list[i]
                output = self.__hidden_n_activation_func(subtotal)
                current_layer_outputs.append(output)

            output_subtotal = np.dot(output, self.__output_w_list) + self.__output_b_list
            batch_outputs.append(self.__output_n_activation_func(output_subtotal))

        return np.asarray(batch_outputs, dtype=np.float64)

    def backward(self, batch_errors):
        pass

    def __fit_forward(self, batch):
        batch_outputs = []
        batch_extra_output = []

        for _vec in batch:
            output = _vec
            current_layer_outputs = [output]

            for i, weights in enumerate(self.__hidden_w_list):
                subtotal = np.dot(output, weights) + self.__hidden_b_list[i]
                output = self.__hidden_n_activation_func(subtotal)
                current_layer_outputs.append(output)

            output_subtotal = np.dot(output, self.__output_w_list) + self.__output_b_list
            batch_outputs.append(self.__output_n_activation_func(output_subtotal))
            batch_extra_output.append(current_layer_outputs)

        return np.asarray(batch_outputs, dtype=np.float64), batch_extra_output

    def fit(self, train_data, train_answ, batch_size, epochs, learning_rate):

        self.__learning_rate = learning_rate

        for epoch in range(1, epochs + 1):

            epoch_error = []

            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answ = train_answ[indexes]

            for i in range(0, len(train_data), batch_size):
                batch_outputs, batch_extra_output = self.__fit_forward(train_data[i:i + batch_size])
                error = self.__back_propagation(batch_outputs,
                                                batch_extra_output,
                                                train_answ[i:i + batch_size])
                if isinstance(error, list):
                    epoch_error.extend(abs(e) for e in error)
                else:
                    epoch_error.append(abs(error))
            avg_error = np.mean(epoch_error)
            self.__loss_history.append(avg_error)

    # def __back_propagation(self, batch_outputs, batch_extra_output, batch_answ):
    #
    #     batch_answ = np.array(batch_answ, dtype=np.float64).reshape(-1, 1)
    #     errors = batch_outputs - batch_answ
    #     error = np.mean(errors)
    #
    #     deltas = error * self.__output_n_activation_func(batch_outputs, True)
    #     delta = np.mean(deltas)
    #
    #     delta_previous_layer_output = np.array([row[-1] for row in batch_extra_output], dtype=np.float64).mean(axis=0)
    #     self.__output_w_list -= self.__learning_rate * delta * delta_previous_layer_output
    #     self.__output_b_list -= self.__learning_rate * delta
    #     next_deltas = (delta * self.__output_w_list *
    #                    self.__hidden_n_activation_func(delta_previous_layer_output, True))
    #     next_delta = np.mean(next_deltas)
    #
    #     for i in range(len(self.__hidden_w_list) - 1, -1, -1):
    #         delta_previous_layer_output = np.array([row[i] for row in batch_extra_output], dtype=np.float64).mean(
    #             axis=0)
    #         delta_previous_layer_output = delta_previous_layer_output.reshape(-1, 1)
    #
    #         self.__hidden_w_list[i] -= self.__learning_rate * delta_previous_layer_output * next_delta
    #         self.__hidden_b_list[i] -= self.__learning_rate * next_delta
    #         next_deltas = (next_delta * self.__hidden_w_list[i] *
    #                        self.__hidden_n_activation_func(delta_previous_layer_output, True))
    #         next_delta = np.mean(next_deltas)
    #
    #     return errors

    def __back_propagation(self, batch_outputs, batch_extra_output, batch_answ):
        print(batch_answ)
        batch_answ = np.array(batch_answ, dtype=np.float64)
        print(f"batch_outputs.shape: {batch_outputs.shape}")
        errors = -batch_answ * np.log(batch_outputs + 1e-8)
        print(errors)
        error = np.sum(errors, axis=1)
        print(errors)

        # errors = batch_outputs - batch_answ
        # error = np.mean(errors)

        deltas = error * self.__output_n_activation_func(batch_outputs, True)
        delta = np.mean(deltas)

        delta_previous_layer_output = np.array([row[-1] for row in batch_extra_output], dtype=np.float64).mean(axis=0)
        self.__output_w_list -= self.__learning_rate * delta * delta_previous_layer_output
        self.__output_b_list -= self.__learning_rate * delta
        next_deltas = (delta * self.__output_w_list *
                       self.__hidden_n_activation_func(delta_previous_layer_output, True))
        next_delta = np.mean(next_deltas)

        for i in range(len(self.__hidden_w_list) - 1, -1, -1):
            delta_previous_layer_output = np.array([row[i] for row in batch_extra_output], dtype=np.float64).mean(
                axis=0)
            delta_previous_layer_output = delta_previous_layer_output.reshape(-1, 1)

            self.__hidden_w_list[i] -= self.__learning_rate * delta_previous_layer_output * next_delta
            self.__hidden_b_list[i] -= self.__learning_rate * next_delta
            next_deltas = (next_delta * self.__hidden_w_list[i] *
                           self.__hidden_n_activation_func(delta_previous_layer_output, True))
            next_delta = np.mean(next_deltas)

        return errors

    def predict(self, input_data):
        output = input_data
        for i, weights in enumerate(self.__hidden_w_list):
            output = self.__hidden_n_activation_func(np.dot(output, weights) + self.__hidden_b_list[i])
        output = self.__output_n_activation_func(np.dot(output, self.__output_w_list) + self.__output_b_list)
        return round(output[0])

    def evaluate(self, test_data, test_answ, accuracy):
        correct = 0
        for x, y in zip(test_data, test_answ):
            if abs(self.predict(x) - y) <= accuracy:
                correct += 1
        return correct / len(test_data)

    def plot_loss(self):
        plt.plot(self.__loss_history, label='Training Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Error changing')
        plt.legend()
        plt.show()


class DenseLayer(Layer, ABC):

    def __init__(self, inputs_count, neurons_count, activation_func, weights_init_func):
        self.__input_n_count = inputs_count
        self.__weights = weights_init_func((inputs_count, neurons_count))
        self.__biases = np.zeros((1, neurons_count))
        self.__activation_func = activation_func
        self.__input_vector = None
        self.__non_activated_vector = None

    def summary(self):
        print(f"\nDense Layer:")
        print(f"\tWeights shape: {self.__weights.shape}")
        print(f"\tBias shape: {self.__weights.shape}")

    def forward(self, input_vector):
        self.__input_vector = input_vector
        self.__non_activated_vector = np.dot(self.__input_vector, self.__weights) + self.__biases
        return self.__activation_func(self.__non_activated_vector)

    # def backward(self, input_errors, alpha):
    #     delta = input_errors * self.__activation_func(self.__output_vector, True)
    #     self.__weights -= alpha * np.dot(self.__input_vector.T, delta) / delta.shape[0]
    #     self.__biases -= alpha * np.mean(delta, axis=0, keepdims=True)
    #     output_errors = np.dot(delta, self.__weights.T)
    #     return output_errors

    # def backward(self, input_errors, alpha):
    #     delta = input_errors * np.mean(self.__activation_func(self.__output_vector, True), axis=0)
    #     delta = np.mean(delta, axis=0, keepdims=True)
    #     print(delta.shape)
    #     input_vector = np.mean(self.__input_vector, axis=0)
    #     print(input_vector.shape)
    #     input_vector = input_vector.reshape(-1, 1)
    #     print(input_vector.shape)
    #     self.__weights -= alpha * delta * input_vector
    #     self.__biases -= alpha * delta
    #     next_delta = delta * self.__weights
    #     return next_delta

    # def backward(self, input_errors, alpha):
    #     # print(f"input_errors.shape={input_errors.shape}")
    #     # print(f"self.__output_vector.shape={self.__output_vector.shape}")
    #     delta = input_errors * self.__activation_func(self.__output_vector, True)
    #     print(delta)
    #     print(delta.shape)
    #     next_delta = delta
    #     delta = np.mean(delta, axis=0, keepdims=True)
    #     print(delta)
    #     print(delta.shape)
    #     # print(delta.shape)
    #     input_vector = np.mean(self.__input_vector, axis=0)
    #     # print(input_vector.shape)
    #     input_vector = input_vector.reshape(-1, 1)
    #     # print(input_vector.shape)
    #     self.__weights -= alpha * delta * input_vector
    #     self.__biases -= alpha * delta
    #     next_delta = np.dot(next_delta, self.__weights.T)
    #     # next_delta = np.dot(delta, self.__weights.T)
    #     # print(next_delta.shape)
    #     return next_delta

    def backward(self, input_errors, alpha):
        delta = input_errors * self.__activation_func(self.__non_activated_vector, True)
        next_delta = delta
        delta = np.mean(delta, axis=0, keepdims=True)
        input_vector = np.mean(self.__input_vector, axis=0)
        input_vector = input_vector.reshape(-1, 1)
        self.__weights -= alpha * delta * input_vector
        self.__biases -= alpha * delta
        next_delta = np.dot(next_delta, self.__weights.T)
        return next_delta


class Model:

    def __init__(self, layers: List[Layer]):
        self.__layers = layers
        self.__loss_func = None

    def summary(self):
        for layer in self.__layers:
            layer.summary()

    # def fit(self, train_data, train_answ, batch_size, epochs, learning_rate):
    #
    #     for epoch in range(epochs):
    #
    #         indexes = np.random.permutation(len(train_data))
    #         train_data = train_data[indexes]
    #         train_answ = train_answ[indexes]
    #
    #         for i in range(0, len(train_data), batch_size):
    #             batch_data = train_data[i:i + batch_size]
    #             batch_answ = train_answ[i:i + batch_size]
    #             batch_output = batch_data
    #             for layer in self.__layers:
    #                 batch_output = layer.forward(batch_output)
    #             batch_errors = batch_answ.reshape(-1, 1) - batch_output
    #             print(batch_answ.shape)
    #             print(batch_output.shape)
    #             print(batch_errors)
    #             for layer in reversed(self.__layers):
    #                 batch_errors = layer.backward(batch_errors, learning_rate)

    def compile(self, loss_func):
        self.__loss_func = loss_func

    def fit(self, train_data, train_answ, batch_size, epochs, learning_rate):

        for epoch in range(epochs):
            print(f'Epoch №{epoch + 1}')
            indexes = np.random.permutation(len(train_data))
            train_data = train_data[indexes]
            train_answ = train_answ[indexes]

            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_answ = train_answ[i:i + batch_size]
                batch_output = batch_data
                for layer in self.__layers:
                    batch_output = layer.forward(batch_output)

                if self.__loss_func:
                    batch_errors = self.__loss_func(batch_answ.reshape(batch_size, batch_output.shape[1]), batch_output)
                    # print(batch_errors)
                else:
                    batch_errors = batch_answ.reshape(batch_size, batch_output.shape[1]) - batch_output
                # print(batch_answ.shape)
                # print(batch_output.shape)
                for layer in reversed(self.__layers):
                    batch_errors = layer.backward(batch_errors, learning_rate)

    def predict(self, input_data):
        output_data = input_data
        for layer in self.__layers:
            output_data = layer.forward(output_data)
        return output_data

    def evaluate(self, test_data, test_answ):
        correct = 0
        if self.__loss_func is None:
            for x, y in zip(test_data, test_answ):
                if round(self.predict(x)[0][0]) == y:
                    correct += 1
            return correct / len(test_data)
        elif self.__loss_func.__name__ == 'cross_entropy':
            for x, y in zip(test_data, test_answ):
                if np.argmax(self.predict(x.reshape(1, -1))) == np.argmax(y):
                    correct += 1
            return correct / len(test_data)

