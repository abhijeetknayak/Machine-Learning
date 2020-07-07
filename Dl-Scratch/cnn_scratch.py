import numpy as np
from cifar import extract_data

class CNN:
    def __init__(self, x, y, conv_params):
        self.input = x
        self.target = y
        self.W = np.random.randn(3, 3, 3)
        self.b = None
        self.stride = conv_params['stride']
        self.pad = conv_params['padding']

    def forward_naive(self):
        N, C, H, W = self.input.shape
        F, HH, WW = self.W.shape
        s, p = self.stride, self.pad

        H_out = (H + 2 * p - HH) // s + 1
        W_out = (W + 2 * p - WW) // s + 1

        pad_width = ((0, ), (0, ), (p, ), (p, ))
        x_pad = np.pad(self.input, pad_width, 'constant')
        output = np.zeros((N, C, H_out, W_out))

        N_pad, C_pad, H_pad, W_pad = x_pad.shape
        print(N_pad, C_pad, H_pad, W_pad)
        print(H_out, W_out)

        for n in range(N):
            for f in range(F):
                for h in range(0, H_pad - HH + 1, s):
                    for w in range(0, W_pad - WW + 1, s):
                        output[n, f, int(h//s), int(w//s)] =\
                            np.sum(np.multiply(x_pad[n, :, h:h+HH, w:w+WW], self.W[f, ])) + self.b[f]

x, x1, y, y1 = extract_data()
params = {'padding': 1, 'stride': 1}
c = CNN(x, y, params)
c.forward_naive()



