"""
Created by Edward Li at 8/23/20
"""


class Activation:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def activation_fn(self, inputs):
        raise NotImplementedError("activation not implemented")

    def activation_prime(self, inputs):
        raise NotImplementedError("activation inverse not implemented")

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.activation_fn(inputs)
        return self.outputs

    def backward(self, output_error):
        return self.activation_prime(self.inputs) * output_error
