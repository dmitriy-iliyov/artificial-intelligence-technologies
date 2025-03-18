import numpy as np

from tools import activation_func
import perceptrons

# perceptron = perceptrons.NonLearnableSLP([1, 1], -1.5, activation_func.activation_threshold)
# perceptron.summary()
# _input = [0, 1]
# output = perceptron.involve(_input)
# print("input =", _input)
# print("output =", output)

train_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

train_answ = np.array([0, 0, 0, 1])

test_data = train_data
test_answ = train_answ

learnable_p = perceptrons.LearnableSLP(activation_func.activation_elu)
learnable_p.summary()
learnable_p.fit(train_data, train_answ)
learnable_p.summary()
print(learnable_p.evaluate(test_data, test_answ))
print(learnable_p.predict([1, 0]))
