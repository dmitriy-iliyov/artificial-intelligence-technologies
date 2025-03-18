import cnn
from tools import weights_init, activation_func


conv_layer = cnn.ConvolutionLayer(
    10, 3, (5, 5),
    weights_init.default_init, activation_func.activation_prelu, 1)

conv_layer.summary()
