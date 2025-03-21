import numpy as np

from lab_7 import cnn

image = np.random.rand(28, 28, 1)

leNet = cnn.LeNet()
leNet.predict(image)
