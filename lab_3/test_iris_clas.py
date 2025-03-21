from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from lab_2.perceptrons import Model, DenseLayer
from tools import activation_func as af, weights_init as wi, loss_func as lf
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

lb = LabelBinarizer()
y_one_hot = lb.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

new_gen_mlp = Model([
    DenseLayer(4, 10, af.tanh, wi.default_init),
    DenseLayer(10, 3, af.softmax, wi.default_init),
])

new_gen_mlp.compile(lf.cross_entropy)
new_gen_mlp.fit(X_train, y_train, 1, 200, 0.05)

accuracy = new_gen_mlp.evaluate(X_test, y_test)
print('accuracy =', accuracy)

test_input = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float64)
prediction = new_gen_mlp.predict(test_input)
print(f'predict {test_input} =', lb.inverse_transform(prediction)[0])