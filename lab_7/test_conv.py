import cnn

import numpy as np
import tensorflow as tf

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

filter_kernel = [
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
]

conv_layer = cnn.ConvolutionLayer(filters_count=1, kernel_shape=(3, 3), stride=1, n_activation_func=lambda x: x,
                                  channels_count=3)

output = conv_layer.forward(image)

print("Output of convolution:")
print(output)

image = np.stack([red_channel, green_channel, blue_channel], axis=-1).astype(np.float32)
image = image[np.newaxis, ...]

filter_weights = np.array([
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]]).astype(np.float32)

filter_weights = filter_weights[..., np.newaxis]

image_tensor = tf.constant(image)
filter_tensor = tf.constant(filter_weights)

output = tf.nn.conv2d(image_tensor, filter_tensor, strides=[1, 1, 1, 1], padding="VALID")

print("Output shape:", output.shape)
print("Output values:\n", output.numpy().squeeze())
