import numpy as np

from lab_7 import cnn


red_channel = np.array([
    [45, 200,  56, 115,  10],
    [249,  44, 111, 124,  58],
    [63, 158,  60, 221,  92],
    [136,  67, 220, 204, 245],
    [105,  87,  55,  48, 167]
])

green_channel = np.array([
    [132,  38,  70, 144, 212],
    [141,  73,  12, 125,  82],
    [71, 122,  60, 107,  72],
    [66,  94,  95, 153, 142],
    [125, 143,  62,  23,  81]
])

blue_channel = np.array([
    [50,  94, 136, 209, 137],
    [71,  38, 249, 158, 143],
    [32,  72,  77, 164, 109],
    [58, 105, 121,  40, 232],
    [15, 102,  58,  68,  73]
])

image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

flatten = cnn.FlattenLayer()
output = flatten.forward(image)
output = flatten.backward(output, )
print(output)
print(output.shape)
