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
        for f in range(n_f):
            # vertically
            for j in range(out_h):
                # horizontally
                for i in range(out_w):
                    output[i, j, f] = np.sum(padded[j * s:j * s + k, i * s:i:s + k, :] * self.filters[f, :]) + \
                                      self.bias[f]

        self.inputs = padded
        self.outputs = output

        return output

    def backward(self, error):
        """
        backward pass
        """

        h, w, c = self.inputs.shape
        h_o, w_o, c_o = self.outputs.shape
        n_f, k, _, n_in = self.filters.shape
        s = self.stride
        # dE/dW = X (x) Y, input convolved with output error, gives us our kernel weight gradients
        dW = np.zeros_like(self.filters)

        for f in range(n_f):
            for j in range(h_o):
                for i in range(w_o):
                    for m in range(k):
                        for n in range(k):
                            for cc in range(c):
                                dW[f,m,n,cc] += error[j,i,f] * self.inputs[s*j+m,s*i+n,cc]


        # dE/dX = Y (x) W', error convolved with our transpose weights
        dX = np.zeros_like(self.inputs)
        dXpad = np.pad(dX,(0,0,self.padding,self.padding), mode="constant")
        
        for j in range(h_o):
            for i in range(w_o):
                for m in range(k):
                    for n in range(k):
                        for f in range(n_f):
                            for cc in range(c):
                                dXpad[s*j+m,s*i+n,cc] += error[j,i,f] * w[f,m,n,cc]

        if self.padding > 0 :
            dX = dXpad[self.padding:-self.padding,self.padding:-self.padding,:]
        else:
            dX = dXpad
            

        self.gradients["input_grad"] = dX
        self.gradients["weight_grad"] = dW
        # dE/dbj = SUM(Yj) (bias gradient is just output gradient)
        self.gradients["bias_grad"] = np.sum(error, axis=[0, 1])

        return dX
