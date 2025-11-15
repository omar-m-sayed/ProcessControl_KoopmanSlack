import numpy as np
from scipy.integrate import solve_ivp


class QuadrupleSim:
    def __init__(self, state_min, state_max, action_min, action_max):
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max
        self.Poly = quadruple(1)
        pass

    def get_real_state_action(self, state, action):
        state = state * (self.state_max - self.state_min) + self.state_min
        action = action * (self.action_max - self.action_min) + self.action_min
        return state, action

    def get_real_state(self, state):
        state = state * (self.state_max - self.state_min) + self.state_min
        return state

    def get_real_action(self, action):
        action = action * (self.action_max - self.action_min) + self.action_min
        return action

    def get_regularized_state(self, state):
        state = (state - self.state_min) / (self.state_max - self.state_min)
        return state

    def get_trajectory(self, state, action, real_action=False):
        real_state = self.get_real_state(state)
        if real_action:
            real_action = action
        else:
            real_action = self.get_real_action(action)
        real_state = self.Poly.get_trajectory(real_state, real_action)
        real_state = self.get_regularized_state(real_state)
        return real_state


class quadruple:
    def __init__(self, dt=1):
        self.dt = dt
        self.A1 = 28
        self.A3 = 28
        self.A2 = 32
        self.A4 = 32
        self.a1 = 0.071
        self.a3 = 0.071
        self.a2 = 0.057
        self.a4 = 0.057
        self.kc = 0.5
        self.g = 981
        self.k1 = 3.33
        self.k2 = 3.35
        self.gamma_1 = 0.7
        self.gamma_2 = 0.6

    def func(self, t, x, u):
        h_1 = x[0]
        h_2 = x[1]
        h_3 = x[2]
        h_4 = x[3]
        v1 = u[0]
        v2 = u[1]

        dh1 = -(self.a1 / self.A1) * np.sqrt(2 * self.g * h_1) + (self.a3 / self.A1) * np.sqrt(
            2 * self.g * h_3) + ((self.gamma_1 * self.k1) / self.A1) * v1
        dh2 = -(self.a2 / self.A2) * np.sqrt(2 * self.g * h_2) + (self.a4 / self.A2) * np.sqrt(
            2 * self.g * h_4) + ((self.gamma_2 * self.k2) / self.A2) * v2

        dh3 = -(self.a3 / self.A3) * np.sqrt(2 * self.g * h_3) + (((1 - self.gamma_2) * self.k2) / self.A3) * v2

        dh4 = -(self.a4 / self.A4) * np.sqrt(2 * self.g * h_4) + (((1 - self.gamma_1) * self.k1) / self.A4) * v1
        return [dh1, dh2, dh3, dh4]

    def ode_step(self, state, u):
        sol = solve_ivp(self.func, [0, self.dt], state, args=(u,), method='RK45', t_eval=[self.dt], rtol=1e-10,
                        atol=1e-10)
        return sol.y[:, -1]


    def get_trajectory(self, x_0, u_traj):
        traj = [x_0]
        for i in range(u_traj.shape[0]):
            traj.append(self.ode_step(traj[-1].reshape([-1]), u_traj[i].reshape([-1])))
        return np.array(traj)

