import numpy as np
import tensorflow_datasets as tfds
from lab_2.perceptrons import Model, BinaryMLP, DenseLayer
from lab_7.cnn import FlattenLayer
from tools import activation_func as af, weights_init as wi, loss_func as lf


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

mlp = Model([
    FlattenLayer(),
    DenseLayer(784, 128, af.relu, wi.default_init),
    DenseLayer(128, 10, af.softmax, wi.default_init)
])
mlp.compile(lf.cross_entropy)
mlp.fit(train_data, train_labels, epochs=5, batch_size=32, learning_rate=0.01)
print(mlp.evaluate(test_data, test_labels))
print(np.argmax(test_labels[1]))
print(np.argmax(mlp.predict(test_data[1].reshape(1, -1))[0]))
