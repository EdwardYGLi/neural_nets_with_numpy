"""
Created by Edward Li at 8/23/20
"""


class LossFunction:
    def loss_fn(self, pred, targ):
        raise NotImplementedError("loss function not implemented")

    def loss_fn_prime(self, pred, targ):
        raise NotImplementedError("loss function inverse not implemented")

