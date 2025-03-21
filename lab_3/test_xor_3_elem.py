import numpy as np
from lab_2.perceptrons import Model, DenseLayer
from tools import activation_func as af, weights_init as wi


train_data = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=np.float64)

train_answ = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float64)

test_data = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=np.float64)

test_answ = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float64)

new_gen_mlp = Model([
    DenseLayer(3, 4, af.tanh, wi.default_init),
    DenseLayer(4, 1, af.sigmoid, wi.default_init),
])

new_gen_mlp.fit(train_data, train_answ, 1, 100, 0.05)

accuracy = new_gen_mlp.evaluate(test_data, test_answ)
print('accuracy =', accuracy)

test_input = np.array([1, 0, 1], dtype=np.float64)
prediction = new_gen_mlp.predict(test_input)
print(f'predict {test_input} =', round(prediction[0][0]))
