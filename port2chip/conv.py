import numpy as np
from functools import reduce
import math

#TODO: rewrite this in Cython
def im2col(image, ksize):
    # NHWC
    i_n = image.shape[1] - ksize + 1
    j_n = image.shape[2] - ksize + 1
    image_col = [1] * (i_n * j_n)
    for i in range(0, i_n):
        for j in range(0, j_n):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col[i * j_n + j] = col
    image_col = np.array(image_col)

    return image_col

class Conv2D:
    def __init__(self, shape, output_channels, ksize=3, method='VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batch_size = shape[0]
        self.ksize = ksize
        self.method = method
        self.eta = np.zeros((shape[0], shape[1], shape[2], self.output_channels))
        self.conv_out = np.zeros(self.eta.shape)

    def set_w_and_b(self, w, b):
        self.weights = w
        self.col_weights = self.weights.reshape([-1, self.output_channels])
        self.bias = b
    
    def forward(self, x):
        if self.method == 'SAME':
            x = np.pad(x, 
                ((0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)), 
                mode='reflect')
        self.col_image = []
        for i in range(self.batch_size):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize)
            self.conv_out[i] = np.reshape(np.dot(self.col_image_i, self.col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return self.conv_out
