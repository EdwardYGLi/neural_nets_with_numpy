"""
Created by Edward Li at 8/23/20
"""
from loss_functions.loss_function import LossFunction
import numpy as np


class MSELoss(LossFunction):
    def loss_fn(self, pred, target):
        # mse = 1/n*SUM(yi-yi')^2
        return np.mean(np.square(pred-target))

    def loss_fn_prime(self, pred, targ):
        # dmse = 2/n * (Yi - Yi')
        return 2*(pred-targ)/pred.size
