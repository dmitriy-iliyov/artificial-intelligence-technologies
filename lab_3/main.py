import numpy as np
# import tensorflow as tf

from lab_2.perceptrons import Model, BinaryMLP, DenseLayer
from tools import activation_func as af, weights_init as wi, loss_func as lf


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

# with BinaryMLP
p = BinaryMLP(4, [4], af.tanh, af.sigmoid)
p.fit(train_data, train_answ, 1, 100, 0.05)
print('accuracy =', p.evaluate(test_data, test_answ))
print('predict', [1, 0, 0, 0], '=', p.predict([1, 0, 0, 0]))
# p.plot_loss()

# with DenseLayer
new_gen_mlp = Model([
    DenseLayer(4, 4, af.tanh, wi.default_init),
    DenseLayer(4, 1, af.sigmoid, wi.default_init),
])
new_gen_mlp.fit(train_data, train_answ, 1, 100, 0.05)
print('accuracy =', new_gen_mlp.evaluate(test_data, test_answ))
print('predict', [1, 0, 0, 0], '=',
      round(new_gen_mlp.predict(np.array([1, 0, 0, 0], dtype=np.float64))[0][0]))

# keras
# model = tf.keras.models.Sequential([
#     tf.keras.Input(shape=(4,)),
#     tf.keras.layers.Dense(4, activation='tanh'),
#     tf.keras.layers.Dense(1, activation='sigmoid'),
# ])
#
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, train_answ, epochs=100, verbose=0)
#
# loss, accuracy = model.evaluate(train_data, train_answ, verbose=0)
# print("accuracy =", accuracy)
# print('predict', [1, 0, 0, 0], '=',
#       round(model.predict(np.array([1, 0, 0, 0], dtype=np.float32).reshape(-1, 4), verbose=0)[0][0]))
