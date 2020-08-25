"""
Created by Edward Li at 8/24/20
"""
import numpy as np

from layers.nn_layer import NNLayer


class Conv2D(NNLayer):
    def __init__(self, in_features, out_features, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=True):
        self.filters = np.random.randn((out_features, kernel_size, kernel_size, in_features), np.float32)
        self.bias = None
        if bias:
            self.bias = np.random.randn(out_features, np.float32)
        self.padding = padding
        self.stride = stride

        # [TODO] implement dilation and groups later
        self.group = groups
        self.dilation = dilation

    def forward(self, inputs):
        n_f, k, _, n_in = self.filters.shape
        h, w, c = inputs.shape

        assert c == n_in, "input channels must equal input feature size"
        # (in_shape−K+2P)/S]+1.
        out_h = ((h - k + 2 * self.padding) / self.stride) + 1
        out_w = ((w - k + 2 * self.padding) / self.stride) + 1

        output = np.zeros((out_h, out_w, n_f), np.float32)

        padded = np.pad(inputs, self.padding, mode="constant")

        s = self.stride
        for f in n_f:
            # vertically
            for j in out_h:
                # horizontally
                for i in out_w:
                    output[i, j, f] = np.sum(padded[j * s:j * s + k, i * s:i:s + k, :] * self.filters[f, :]) + \
                                      self.bias[f]

        self.inputs = padded
        self.outputs = output

        return output

    def backward(self,error):
        """
        backward pass
        """
