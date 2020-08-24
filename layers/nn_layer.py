"""
Created by Edward Li at 8/23/20
"""


# base layer class where all our network layers will inherit from
class NNLayer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.gradients = {}

    # compute forward pass output Y given input tensor X
    def forward(self, inputs):
        raise NotImplementedError("forward not implemented")

    # compute backward pass and derivatives (dE/dX, and dE/dY) and update parameters
    def backward(self, error):
        # should store gradients here
        raise NotImplementedError("backward not implemented")

    # if using default sgd will update weights
    def update_weights(self, learning_rate):
        # should update weights based on learning rate
        raise NotImplementedError("weight update not implemented")
