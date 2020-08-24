"""
Created by Edward Li at 8/23/20
"""
import numpy as np


def Sigmoid(Activation):
    def activation_fn(self, inputs):
        return 1 / (1 + np.exp(-inputs))

    def activation_prime(self, inputs):
        # sig'(x) = e^-x*(1+e^-x)^-2
        # sigmoid derivative = f(x)*(1-f(x))
        return self.activation_fn(inputs) * (1 - self.activation_fn(inputs))
