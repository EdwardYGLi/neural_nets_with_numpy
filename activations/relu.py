"""
Created by Edward Li at 8/23/20
"""
import numpy as np


def ReLU(Activation):
    def activation_fn(self, inputs):
        # ReLU(x) = 0 if x <= 0
        #           x else
        return np.maximum(inputs, 0)

    def activation_prime(self, inputs):
        # dReLU(x) = 0 if x<=0
        #            1 else
        # can use heavyside here
        return np.heavyside(inputs, 0)
