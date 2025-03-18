import numpy as np


class HopfildNN:

    def __init__(self, width, height):
        length = height * width
        self.__weight = np.zeros((length, length))

    def summary(self):
        print(self.__weight)

    def fit(self, train_data):
        for _vec in train_data:
            data_in_vec = np.array(_vec, dtype=np.float64)
            data_in_matrix = data_in_vec * data_in_vec.reshape(-1, 1)
            self.__weight += data_in_matrix
        for i in range(len(self.__weight)):
            self.__weight[i, i] = 0
        self.__weight *= 1/len(train_data)

    def predict(self, input_data):
        output = np.dot(input_data, self.__weight)
        return np.where(output >= 0, 1, -1)

    def evaluate(self, test_data, test_answ):
        correct = 0
        for x, y in zip(test_data, test_answ):
            if abs(sum(self.predict(x)) - sum(y)) == 0:
                correct += 1
        return correct / len(test_data)
