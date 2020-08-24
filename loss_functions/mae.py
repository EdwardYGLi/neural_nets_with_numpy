"""
Created by Edward Li at 8/23/20
"""
from loss_functions.loss_function import LossFunction
import numpy as np


class MSELoss(LossFunction):
    def loss_fn(self, pred, target):
        # mse = 1/n*SUM (abs(yi-yi'))
        return np.mean(np.abs(pred-target))

    def loss_fn_prime(self, pred, targ):
        # dmae = +1 if pred > targ
        #        -1 elif pred < targ
        #     undefined(0 for now)  else
        diff = pred - targ
        diff[diff>0] = 1
        diff[diff<0] = -1
        diff[diff==0] = 0
        return diff
