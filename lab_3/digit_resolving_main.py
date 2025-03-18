import numpy as np
import tensorflow_datasets as tfds
from tools import activation_func
from lab_2 import perceptrons


def load_data():
    (ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)
    return ds_train, ds_test


def prepared_data():
    ds_train, ds_test = load_data()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for image, label in tfds.as_numpy(ds_train):
        x_train.append(image)
        y_train.append(label)
    for image, label in tfds.as_numpy(ds_test):
        x_test.append(image)
        y_test.append(label)

    return (np.expand_dims(np.array(x_train), axis=-1) / 255.0, np.array(y_train),
            np.expand_dims(np.array(x_test), axis=-1) / 255.0, np.array(y_test))


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


def prepared_1d_data():
    ds_train, ds_test = load_data()
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for image, label in tfds.as_numpy(ds_train):
        x_train.append(image.flatten())
        y_train.append(label)

    for image, label in tfds.as_numpy(ds_test):
        x_test.append(image.flatten())
        y_test.append(label)

    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0

    y_train = one_hot_encode(np.array(y_train), num_classes=10)
    y_test = one_hot_encode(np.array(y_test), num_classes=10)

    return x_train, y_train, x_test, y_test


train_data, train_labels, test_data, test_labels = prepared_1d_data()

# model = Model()
# model.fit(train_data, train_labels, epochs=5, batch_size=32)
# model.predict(test_data, test_labels, 10)

perceptron = perceptrons.MultiCategoricalMLP(28 * 28, [128], activation_func.activation_relu, 10, activation_func.activation_sigmoid)
perceptron.fit(train_data, train_labels, 32, 5, 0.001)
perceptron.evaluate(test_data, test_labels, 0.1)
# perceptron.predict()
