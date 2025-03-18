import random

import hopfild_nn
import numpy as np


def vec_to_console(vec, w):
    row = ""
    for i in range(len(vec)):
        c = " "
        if vec[i] == 1:
            c = "*"
        row += f"{c:2} "
        if (i + 1) % w == 0:
            print(row)
            row = ""


def noising(vec, k=0.1):
    noised_cell_count = int(len(vec) * k)
    noised_vec = vec.copy()
    for _ in range(noised_cell_count):
        index = random.randint(0, len(noised_vec) - 1)
        noised_vec[index] *= -1
    return noised_vec


digits = {
    # "0": np.array([
    #     -1, 1, 1, 1, -1,
    #     1, -1, -1, -1, 1,
    #     1, -1, -1, -1, 1,
    #     1, -1, -1, -1, 1,
    #     1, -1, -1, -1, 1,
    #     1, -1, -1, -1, 1,
    #     -1, 1, 1, 1, -1
    # ]),
    "1": np.array([
        -1, -1, 1, -1, -1,
        -1, 1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, 1, 1, -1
    ]),
    "2": np.array([
        -1, 1, 1, 1, -1,
        1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, 1, 1, -1,
        -1, 1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ]),
    "3": np.array([
        -1, 1, 1, 1, -1,
        1, -1, -1, -1, 1,
        -1, -1, -1, -1, 1,
        -1, -1, 1, 1, -1,
        -1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        -1, 1, 1, 1, -1
    ]),
    "4": np.array([
        -1, -1, 1, 1, -1,
        -1, 1, -1, 1, -1,
        1, -1, -1, 1, -1,
        1, -1, -1, 1, -1,
        1, 1, 1, 1, 1,
        -1, -1, -1, 1, -1,
        -1, -1, -1, 1, -1
    ]),
    "5": np.array([
         1,  1,  1,  1,  1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1, -1,
        -1, -1, -1, -1,  1,
        -1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1
    ]),
    "6": np.array([
        -1,  1,  1,  1, -1,
         1, -1, -1, -1, -1,
         1, -1, -1, -1, -1,
         1,  1,  1,  1, -1,
         1, -1, -1, -1,  1,
         1, -1, -1, -1,  1,
        -1,  1,  1,  1, -1
    ])
    # "7": np.array([
    #      1,  1,  1,  1,  1,
    #     -1, -1, -1, -1,  1,
    #     -1, -1, -1,  1, -1,
    #     -1, -1,  1, -1, -1,
    #     -1,  1, -1, -1, -1,
    #     -1,  1, -1, -1, -1,
    #     -1,  1, -1, -1, -1
    # ]),
    # "8": np.array([
    #     -1,  1,  1,  1, -1,
    #      1, -1, -1, -1,  1,
    #      1, -1, -1, -1,  1,
    #     -1,  1,  1,  1, -1,
    #      1, -1, -1, -1,  1,
    #      1, -1, -1, -1,  1,
    #     -1,  1,  1,  1, -1
    # ]),
    # "9": np.array([
    #     -1,  1,  1,  1, -1,
    #      1, -1, -1, -1,  1,
    #      1, -1, -1, -1,  1,
    #     -1,  1,  1,  1,  1,
    #     -1, -1, -1, -1,  1,
    #     -1, -1, -1, -1,  1,
    #     -1,  1,  1,  1, -1
    # ])
}

p = hopfild_nn.HopfildNN(5, 7)
p.fit(digits.values())

test = digits["6"]
print("Digit:")
vec_to_console(test, 5)
test = noising(test, 0.1)
print("Noised digit:")
vec_to_console(test, 5)
print("Predicted digit:")
predicted = p.predict(test)
vec_to_console(predicted, 5)

evaluated = p.evaluate([noising(_, 0.2) for _ in digits.values()], digits.values())
print(evaluated)
