import torch.nn as nn
import casadi as ca
from casadi.tools import struct_symSX, entry, DM, horzcat, Function, SX, jacobian, MX, struct_symMX, vertcat
"""
Optimization process to recover the orignal state from the slack states. 

"""

class inverseOpt:
    def __init__(self,num_states,horizion,NN_encoder,translate_net,scale_net):
        self.nx = num_states
        self.T = horizion
        self.encoder_weight, self.encoder_bias = self.get_weight_bias(NN_encoder)
        self.trans_weight, self.trans_bias = self.get_weight_bias(translate_net)
        self.scale_weight, self.scale_bias = self.get_weight_bias(scale_net)

        self.x = ca.MX.sym("X", self.nx, self.T)
        self.s_input = ca.MX.sym("S_in", self.nx, self.T)
        rows = [self.masked_row_expr(i) for i in range(self.nx)]
        S_out = ca.vertcat(*rows)  # (nx, T)
        S = ca.Function("S", [self.x, self.s_input], [S_out])

        self.opt_x = struct_symMX([
            entry('s', shape=(self.nx, self.T)),
            entry('x', shape=(self.nx, self.T)),
        ])

        g_expr = self.opt_x['x'] - S(self.opt_x['x'], self.s_input)
        g = ca.reshape(g_expr, (-1, 1))

        J = ca.MX.zeros(1)

        p = ca.reshape(self.s_input, (-1, 1))

        prob = {
            'f': J,
            'x': self.opt_x.cat,
            'g': g,
            'p': p
        }
        self.solver = ca.nlpsol('solver', 'ipopt', prob)

    def get_weight_bias(self,NN):
        weights = []
        bias = []

        for i in NN:
            if isinstance(i, nn.Linear):
                weights.append(i.weight.detach().numpy())
                bias.append(i.bias.detach().numpy())

        return weights, bias

    def nn_function(self,x, weigts, bias):
        for i in range(len(weigts) - 1):
            x = weigts[i] @ x + bias[i]
            x = ca.tanh(x )  # elementwise max(0, x) → ReLU
        x = weigts[-1] @ x + bias[-1]
        return x

    def masked_row_expr(self,i):
        """
        Build (s_input - t(x_masked)) * exp(-s(x_masked)) for the i-th row,
        where x_masked keeps only row i (others are zero).
        Returns a row MX of shape (1, T).
        """
        e_i = ca.DM.eye(self.nx)[:, i]  # (nx,)
        mask_mat = ca.MX(e_i).reshape((self.nx, 1)) @ ca.MX.ones(1, self.T)  # (nx, T)

        x_masked = self.x * mask_mat
        x_in = self.nn_function(x_masked, self.encoder_weight, self.encoder_bias)
        out_t = self.nn_function(x_in, self.trans_weight, self.trans_bias)
        out_s = self.nn_function(x_in, self.scale_weight, self.scale_bias)

        s_expr = (self.s_input)* ca.exp(out_s) + out_t  # (nx, T)
        return s_expr[i, :]  # (1, T)


