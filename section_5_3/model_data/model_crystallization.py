import matplotlib.pyplot as plt
import numpy as np
from casadi import SX, Function, exp, vertcat, jacobian
from scipy.integrate import solve_ivp
import pickle


class CrystallizationSimulate:
    def __init__(self, state_min, state_max, action_min, action_max):
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max
        self.Poly = Crystallization(5)
        pass

    def get_real_state_action(self, state, action):
        state = state * (self.state_max - self.state_min) + self.state_min
        action = action * (self.action_max - self.action_min) + self.action_min
        return state, action

    def get_real_state(self, state, axis=None):
        if axis is None:
            # Apply scaling to the whole state vector
            return state * (self.state_max - self.state_min) + self.state_min
        else:
            # Apply scaling only along the specified axis
            return state * (self.state_max[axis] - self.state_min[axis]) + self.state_min[axis]

    def get_real_action(self, action):
        action = action * (self.action_max - self.action_min) + self.action_min
        return action

    def get_regularized_state(self, state):
        state = (state - self.state_min) / (self.state_max - self.state_min)
        return state

    def get_trajectory(self, state, action, real_action=False):
        real_state = self.get_real_state(state)[:6]
        if real_action:
            real_action = action
        else:
            real_action = self.get_real_action(action)
        real_state = self.Poly.get_trajectory(real_state, real_action)
        real_state = np.hstack([real_state, (real_state[:, 4] / real_state[:, 3]).reshape([-1, 1])])
        real_state = self.get_regularized_state(real_state)
        return real_state


class Crystallization:
    def __init__(self, dt=5):
        self.dt = dt

        self.nx = 6
        self.nu = 2
        self.x = SX.sym('x', self.nx)
        self.u = SX.sym('u', self.nu)

        self.rho = 1043
        self.cp = 4

        c = self.x[0]
        t = self.x[1]
        t_j = self.x[2]
        mu_0 = self.x[3]
        mu_1 = self.x[4]
        mu_2 = self.x[5]

        T_j_in = 300
        F_j = self.u[0]
        F_feed = self.u[1]

        self.T_feed = 350
        self.C_feed = 0.11238 * np.exp(9.0849e-3 * (self.T_feed - 273.15))
        self.A = 10
        self.U = 1000
        self.cpj = 4
        self.V = 10
        self.m = self.rho * self.V
        self.rhoj = 1050
        self.mu_0_in = 1.0292 * 1e8
        self.mu_1_in = 4.117 * 1e4
        self.mu_2_in = 1.7501 * 1e1
        self.rho_cryst = 1432
        self.delta_H_cyst = 0
        self.k_v_cryst = np.pi / 6
        self.VJ = 10
        self.m_j = self.rhoj * self.VJ
        c_star = 0.11238 * np.exp(9.0849e-3 * (t - 273.15))
        delta_s = (c - c_star) / c_star
        G = 5.857 * 1e-5 * delta_s ** 2 * np.tanh(0.9113 / delta_s)
        m_dot_crystal = 3 * self.V * self.k_v_cryst * self.rho_cryst * G * mu_2
        dot_c = (1 / self.m) * (-m_dot_crystal + self.rho * F_feed * (self.C_feed - c))
        dot_T = (1 / (self.m * self.cp)) * (-self.delta_H_cyst * m_dot_crystal + self.rho * F_feed * self.cp * (
                self.T_feed - t) - self.U * self.A * (t - t_j))
        dot_t_j = (1 / (self.m_j * self.cpj)) * (
                self.rhoj * F_j * self.cp * (T_j_in - t_j) - self.U * self.A * (t_j - t))
        d_mu_0 = ((self.rho * F_feed) / self.m) * (self.mu_0_in - mu_0) + + (c - c_star) * 1e5
        d_mu_1 = G * mu_0 + ((self.rho * F_feed) / self.m) * (self.mu_1_in - mu_1)
        d_mu_2 = 2 * G * mu_1 + ((self.rho * F_feed) / self.m) * (self.mu_2_in - mu_2)

        # self.xdot = vertcat(dot_c, dot_T, dot_t_j, d_mu_0, d_mu_1, d_mu_2)
        # self.system = Function("sys", [self.x, self.u], [self.xdot])


    def func(self, t, x, u):
        c = x[0]
        t = x[1]
        t_j = x[2]
        mu_0 = x[3]
        mu_1 = x[4]
        mu_2 = x[5]

        T_j_in = 300
        F_j = u[0]
        F_feed = u[1]
        c_star = 0.11238 * np.exp(9.0849e-3 * (t - 273.15))
        delta_s = (c - c_star) / c_star

        G = 5.857 * 1e-5 * delta_s ** 2 * np.tanh(0.9113 / delta_s)
        m_dot_crystal = 3 * self.V * self.k_v_cryst * self.rho_cryst * G * mu_2

        dot_c = (1 / self.m) * (-m_dot_crystal + self.rho * F_feed * (self.C_feed - c))
        dot_T = (1 / (self.m * self.cp)) * (-self.delta_H_cyst * m_dot_crystal + self.rho * F_feed * self.cp * (
                self.T_feed - t) - self.U * self.A * (t - t_j))
        dot_t_j = (1 / (self.m_j * self.cpj)) * (
                self.rhoj * F_j * self.cp * (T_j_in - t_j) - self.U * self.A * (t_j - t))
        d_mu_0 = ((self.rho * F_feed) / self.m) * (self.mu_0_in - mu_0) + + (c - c_star) * 1e5
        d_mu_1 = G * mu_0 + ((self.rho * F_feed) / self.m) * (self.mu_1_in - mu_1)
        d_mu_2 = 2 * G * mu_1 + ((self.rho * F_feed) / self.m) * (self.mu_2_in - mu_2)
        return [dot_c, dot_T, dot_t_j, d_mu_0, d_mu_1, d_mu_2]

    def ode_step(self, state, u):
        sol = solve_ivp(self.func, [0, self.dt], state, args=(u,), method='RK45', t_eval=[self.dt], rtol=1e-10,
                        atol=1e-10)
        return sol.y[:, -1]
    def get_trajectory(self, x_0, u_traj):
        traj = [x_0.reshape([-1])]
        for i in range(u_traj.shape[0]):
            traj.append(self.ode_step(traj[-1].reshape([-1]), u_traj[i].reshape([-1])))
        return np.array(traj)
