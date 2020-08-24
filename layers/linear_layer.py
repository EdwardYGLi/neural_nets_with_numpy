"""
Created by Edward Li at 8/23/20
"""
import numpy as np


def Linear(NNLayer):
    def __init__(self, in_features, out_features, with_bias=True):
        """
        Define a linear layer
        :param in_features: number of input features
        :param out_features: number of output features
        :param with_bias: add bias, default = True.
        :return: Create a linear layer
        """
        self.weights = np.random.randn((in_features, out_features), np.float32)
        self.bias = None
        if with_bias:
            self.bias = np.random.randn((1, out_features), np.float32)

    def forward(self, input_tensor):
        self.inputs = input_tensor
        self.output = np.dot(self.inputs, self.weights)
        # add bias if specified
        if self.bias is not None:
            self.output += self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # dE/dW = [dE/dw11 ..............dE/dw1j,
        #          ....                     ....,
        #          dE/dwi1 ..............dE/dwij]
        # dE/dwij = dE/y1*dy1/dwij + ... + dE/dyj* dyj/dwij = dE/dyj * xi
        # dE/dW = x_transpose * dE/dY
        self.gradients["weight_grad"] = np.dot(self.input.T, output_error)

        # dE/dB = [dE/db1 ....... dE/dbj]
        # dE/dbj = dE/dyj (bias gradient is just output gradient)
        self.gradients["bias_grad"] = output_error

        # dE/dX = [dE/dx1 ........ dE/dxj]
        # chain rule
        # dE/dxi = dE/dy1*dy1/dx1 + ....  + dE/dyj* dyj/dxi
        # because dyj/dxi = wij (linear layer)
        # dE/dxi = dE/dy1 * wi1 +  .... + dE/dyj * wij
        # dE/dX = dE/dY * wT
        self.gradients["input_grad"] = np.dot(output_error, self.weights.T)

        return self.gradients["input_grad"]

    def update_weights(self, learning_rate):
        # update learning rates
        self.weights -= learning_rate * self.gradients["weights_grad"]
        self.bias -= learning_rate * self.gradients["bias_grad"]
