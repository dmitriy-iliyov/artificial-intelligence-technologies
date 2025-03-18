import numpy as np
from matplotlib import pyplot as plt

from tools import weights_init as wi, activation_func as af


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
        return af.activation_threshold(np.dot(input_data, self.__weights) + self.__b)

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

    def __deprecated__back_propagation(self, batch_outputs, batch_extra_output, batch_answ):

        batch_answ = np.array(batch_answ, dtype=np.float64).reshape(-1, 1)
        errors = batch_outputs - batch_answ
        deltas = errors * self.__output_n_activation_func(batch_outputs, True)

        for _di, delta in enumerate(deltas):
            self.__output_w_list -= self.__learning_rate * delta * batch_extra_output[_di][-1]
            self.__output_b_list -= self.__learning_rate * delta
            next_deltas = (delta * self.__output_w_list *
                           self.__hidden_n_activation_func(batch_extra_output[_di][-1], True))

            for i in range(len(self.__hidden_w_list) - 1, -1, -1):
                self.__hidden_w_list[i] -= self.__learning_rate * np.outer(batch_extra_output[_di][i], next_deltas)
                self.__hidden_b_list[i] -= self.__learning_rate * next_deltas.reshape(-1)
                next_deltas = (np.dot(self.__hidden_w_list[i], next_deltas) *
                               self.__hidden_n_activation_func(batch_extra_output[_di][i], True))

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


class MultiCategoricalMLP:

    def __init__(self, input_n_count, hidden_layers, hidden_n_activation_func, output_n_count, output_n_activation_func):

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
