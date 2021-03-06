"""Model trained on MNIST dataset with variable input shape."""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.layers import (Conv2D, GlobalAveragePooling2D,
                                     MaxPooling2D, Input)
from tensorflow.keras.models import Model  # , Sequential
# from utils.exceptions import TooSmallInputShape

(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_train = x_train / 255

# def nin_block(filters, kernel_size, strides, padding, inputs=None):
#     if inputs:
#         inp = Input(inputs)
#         x = Conv2D(filters, kernel_size, strides=strides,
#                    padding=padding, activation="relu")(inp)
#     else:
#         inp = Conv2D(filters, kernel_size, strides=strides,
#                      padding=padding, activation="relu")
#         x = inp

#     x = Conv2D(filters, kernel_size=1, activation="relu")(x)
#     x = Conv2D(filters, kernel_size=1, activation="relu")(x)
#     pool = MaxPooling2D(pool_size=(2, 2))(x)
#     return Model(inputs=inp, outputs=pool)


class NinBlock(tf.keras.layers.Layer):
    """Conv2D 3x3 -> 1x1 -> 1x1."""

    def __init__(self, filters, kernel_size, strides, padding, inputs=None):
        """Initialize layers."""
        super(NinBlock, self).__init__()
        # self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        self.conv1 = Conv2D(self.filters, self.kernel_size, strides=self.strides,
                            padding=self.padding, activation="relu")
        self.conv2 = Conv2D(self.filters, 1, activation="relu")
        self.conv3 = Conv2D(self.filters, 1, activation="relu")
        self.max_pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, input_tensor):
        """Connect layers."""
        inp = self.conv1(input_tensor)
        x = self.conv2(inp)
        x = self.conv3(x)
        return self.max_pool(x)


class NinMnist(Model):
    """
    Model based on network in network architecture.
    It is adjusted for mnist dataset.
    """
    def __init__(self, inputs=[28, 28, 1]):
        super(NinMnist, self).__init__()
        self.inputs = inputs
        # self.minimum_input_shape = (14, 14, 1)

        self.block1 = NinBlock(16, 3, 1, "same")
        self.block2 = NinBlock(32, 3, 1, "same")
        self.block3 = NinBlock(64, 3, 1, "same")
        self.conv2d = Conv2D(filters=10, kernel_size=(3, 3), strides=1)
        self.global_pool = GlobalAveragePooling2D()

    def call(self, input_tensor):
        x = self.block1(input_tensor)
        x = self.block2(x)
        x = self.block3(x)
        x = self.conv2d(x)
        return self.global_pool(x)

    def model(self):
        inp = Input(self.inputs)
        return Model(inputs=inp, outputs=self.call(inp))

    # def build(self, input_shape):

    # #     width, height, channels = self.inputs
    # #     if (self.minimum_input_shape[0] > width or
    # #         self.minimum_input_shape[1] > height or
    # #         self.minimum_input_shape[2] > channels):
    # #         raise TooSmallInputShape(self.inputs, self.minimum_input_shape)
    #     inp = Input(shape=self.inputs)
    #     self.block1(inp)


model = NinMnist([56, 56, 1])
model.build([28, 28, 1])
# X = np.random.uniform(size=(1, 28, 28, 1))
# for layer in model.layers:
#     X = layer(X)
#     print(layer.name, 'output shape:\t', X.shape)
# model.summary()

# print(model.output)
# model = Sequential()
# model.add(Input([28, 28, 1]))
# model.add(NinBlock(16, 3, 1, "same"))
# # model.add(NinBlock(32, 3, 1, "same"))
# # model.add(NinBlock(64, 3, 1, "same"))
# model.add(Conv2D(10, kernel_size=3))
# model.add(GlobalAveragePooling2D())

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer="adam",
              metrics=["accuracy"])
# model.build([None, 28, 28, 1])
model.model().summary()
print(model.output)
print(model.input)
model.fit(x_train, y_train, epochs=1)
