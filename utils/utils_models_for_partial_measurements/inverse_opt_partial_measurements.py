import torch.nn as nn
import casadi as ca
from casadi.tools import struct_symSX, entry, DM, horzcat, Function, SX, jacobian, MX, struct_symMX, vertcat

"""
Optimization process to recover the orignal state from the slack states. 

"""


class inverseOpt:
    def __init__(self, num_states, horizion, NN_encoder, translate_net, scale_net, state_delay, real_states):
        self.nx = num_states
        self.T = horizion
        self.state_delay = state_delay
        self.real_states = real_states

        self.encoder_weight, self.encoder_bias = self.get_weight_bias(NN_encoder)
        self.trans_weight, self.trans_bias = self.get_weight_bias(translate_net)
        self.scale_weight, self.scale_bias = self.get_weight_bias(scale_net)

        self.x = ca.MX.sym("X", self.nx, self.T)
        self.s_input = ca.MX.sym("S_in", self.real_states, self.T)
        rows = [self.masked_row_expr(i) for i in range(self.real_states)]
        S_out = ca.vertcat(*rows)  # (nx, T)
        S = ca.Function("S", [self.x, self.s_input], [S_out])

        self.opt_x = struct_symMX([
            entry('s', shape=(self.nx, self.T)),
            entry('x', shape=(self.nx, self.T)),
        ])

        g_expr = self.opt_x['x'][self.real_states * (state_delay):self.real_states * (state_delay + 1), :] - S(
            self.opt_x['x'], self.s_input)
        g = ca.reshape(g_expr, (-1, 1))

        J = ca.MX.zeros(1)
        J = 1e-5 * ca.sumsqr(self.opt_x['x'])

        p = ca.reshape(self.s_input, (-1, 1))

        prob = {
            'f': J,
            'x': self.opt_x.cat,
            'g': g,
            'p': p
        }
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_constr_viol_tol': 1e-5,
            'ipopt.max_iter': 500,
            'ipopt.hessian_approximation': 'limited-memory',  # robust start
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.bound_relax_factor': 1e-8,
            'ipopt.bound_push': 1e-8,
            'ipopt.bound_frac': 1e-8,
            'ipopt.linear_solver': 'mumps',  # or 'ma57' if available
        }
        # solver = ca.nlpsol('solver', 'ipopt', prob, opts)
        self.solver = ca.nlpsol('solver', 'ipopt', prob)

    def get_weight_bias(self, NN):
        weights = []
        bias = []

        for i in NN:
            if isinstance(i, nn.Linear):
                weights.append(i.weight.detach().numpy())
                bias.append(i.bias.detach().numpy())

        return weights, bias

    def nn_function(self, x, weigts, bias):
        for i in range(len(weigts) - 1):
            x = weigts[i] @ x + bias[i]
            x = ca.tanh(x )
        x = weigts[-1] @ x + bias[-1]
        return x

    def nn_function_no(self, x, weigts, bias):
        for i in range(len(weigts) - 1):
            x = weigts[i] @ x + bias[i]
            x = ca.tanh(x)
        # # elementwise max(0, x) → ReLU
        x = weigts[-1] @ x + bias[-1]
        return x

    def masked_row_expr(self, i):
        """
        Build (s_input - t(x_masked)) * exp(-s(x_masked)) for the i-th row,
        where x_masked keeps only row i (others are zero).
        Returns a row MX of shape (1, T).
        """
        e_i = ca.DM.eye(self.nx)[:, i + self.real_states * self.state_delay]  # (nx,)
        mask_mat = ca.MX(e_i).reshape((self.nx, 1)) @ ca.MX.ones(1, self.T)  # (nx, T)

        x_masked = self.x * mask_mat
        x_in = self.nn_function(x_masked, self.encoder_weight, self.encoder_bias)
        out_t = self.nn_function_no(x_in, self.trans_weight, self.trans_bias)
        out_s = self.nn_function_no(x_in, self.scale_weight, self.scale_bias)

        s_expr = (self.s_input) * ca.exp(out_s) + out_t  # (nx, T)
        return s_expr[i, :]  # (1, T)
