"""
Created by Edward Li at 8/23/20
"""
import numpy as np


def Tanh(Activation):
    def activation_fn(self, inputs):
        return np.tanh(inputs)

    def activation_prime(self, inputs):
        return 1 - np.tanh(inputs) ** 2
