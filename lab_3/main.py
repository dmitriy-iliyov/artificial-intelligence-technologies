import numpy as np

from lab_2 import perceptrons
from tools import activation_func as af


def prepare_data_for_4x_xor(split=True):
    variables = np.array(np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1])).T.reshape(-1, 4)

    xor_result = np.bitwise_xor(np.bitwise_xor(variables[:, 0], variables[:, 1]),
                                np.bitwise_xor(variables[:, 2], variables[:, 3]))
    if not split:
        return variables, xor_result

    return np.array(variables[:12]), np.array(xor_result[:12]), np.array(variables[12:]), np.array(xor_result[12:])


# train_data = np.array([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])
#
# train_answ = np.array([0, 1, 1, 1])
#
#

# train_data, train_answ, test_data, test_answ = prepare_data_for_4x_xor(True)

train_data, train_answ = prepare_data_for_4x_xor(False)
test_data = train_data
test_answ = train_answ


p = perceptrons.BinaryMLP(4, [32, 16, 4], af.activation_tanh, af.activation_sigmoid)
p.summary()
p.fit(train_data, train_answ, 1, 100, 0.01)
print('evaluate =', p.evaluate(test_data, test_answ, 0.2))
print('predict', [1, 0, 0, 0], '=', p.predict([1, 0, 0, 0]))
p.plot_loss()
