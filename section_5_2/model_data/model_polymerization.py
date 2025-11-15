import numpy as np
from casadi import SX, Function, exp, vertcat,jacobian
from scipy.integrate import solve_ivp



class PolymerizationSimulate:
    def __init__(self, state_min, state_max, action_min, action_max):
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max
        self.Poly = Polymerization(0.03)
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

    def get_trajectory(self, state, action,real_action_flag=False):

        if real_action_flag:
            real_action = action
        else:
            real_action = self.get_real_action(action)

        real_state = self.get_real_state(state)[:4]
        real_state = self.Poly.get_trajectory(real_state, real_action)
        real_state = np.hstack([real_state, (real_state[:, 3] / real_state[:, 2]).reshape([-1, 1])])
        real_state = self.get_regularized_state(real_state)
        return real_state


class Polymerization:
    def __init__(self, dt=0.03):
        self.dt = dt
        self.Zp = 1.77 * 1e9
        self.Ztc = 3.8223 * 1e10
        self.Ep = 1.8283 * 1e4
        self.Ei = 1.255 * 1e5
        self.Etd = 2.9442 * 1e3
        self.Efm = 7.4478 * 1e4
        self.R = 8.314
        self.T = 335
        self.Etc = 2.9442 * 1e3
        self.V = 0.1
        self.C_min = 6
        self.C_in = 8
        self.F = 1
        self.Zfm = 1.0067 * 1e15
        self.Z_p = 1.77 * 1e9

        self.nx = 4
        self.nu = 1

        self.x = SX.sym('x', self.nx)
        self.u = SX.sym('u', self.nu)
        self.f_star = 0.58
        self.Zi = 3.7920 * 1e18
        self.Ztd = 3.1457 * 1e11
        self.M_m = 100.12
        cm = self.x[0]
        ci = self.x[1]
        d0 = self.x[2]
        di = self.x[3]

        f_i = self.u[0]

        P_0 = np.sqrt((2 * self.f_star) * ci * self.Zi * np.exp((-self.Ei) / (self.R * self.T)) / (
                self.Ztd * np.exp((-self.Etd) / (self.R * self.T)) + self.Ztc * np.exp(
            (-self.Etc) / (self.T * self.R))))


        cm_dot =-self.Zi * np.exp((-self.Ei) / (self.R * self.T)) * ci - ((self.F * ci) / self.V) + (
                f_i * self.C_in) / self.V
        ci_dot = -self.Zi * np.exp((-self.Ei) / (self.R * self.T)) * ci - ((self.F * ci) / self.V) + (
                f_i * self.C_in) / self.V
        d0_dot = (0.5 * self.Ztc * np.exp(-self.Etc / (self.R * self.T)) + self.Ztd * np.exp(
            -self.Etd / (self.R * self.T))) * P_0 ** 2 + self.Zfm * np.exp(-self.Efm / (self.R * self.T)) * cm * P_0 - (
                         self.F * d0) / self.V
        d1_dot = self.M_m * (self.Z_p * np.exp((-self.Ep) / (self.R * self.T)) + self.Zfm * np.exp(
            -self.Efm / (self.R * self.T))) * cm * P_0 - (self.F * di / self.V)

        self.xdot = vertcat(cm_dot, ci_dot, d0_dot, d1_dot)
        self.system = Function("sys", [self.x, self.u], [self.xdot])
        A = jacobian(self.xdot, self.x)
        B = jacobian(self.xdot, self.u)

        self.A_fun = Function('A_fun', [self.x, self.u], [A])
        self.B_fun = Function('B_fun', [self.x, self.u], [B])

    def func(self, t, x, u):
        cm = x[0]
        ci = x[1]
        d0 = x[2]
        di = x[3]
        f_i = u
        P_0 = np.sqrt((2 * self.f_star) * ci * self.Zi * np.exp((-self.Ei) / (self.R * self.T)) / (
                self.Ztd * np.exp((-self.Etd) / (self.R * self.T)) + self.Ztc * np.exp(
            (-self.Etc) / (self.T * self.R))))

        cm_dot = -(self.Zp * np.exp((-self.Ep / (self.R * self.T)))) * cm * P_0 - ((self.F * cm) / self.V) + (
                (self.F * self.C_min) / self.V)
        ci_dot = -self.Zi * np.exp((-self.Ei) / (self.R * self.T)) * ci - ((self.F * ci) / self.V) + (
                f_i * self.C_in) / self.V
        d0_dot = (0.5 * self.Ztc * np.exp(-self.Etc / (self.R * self.T)) + self.Ztd * np.exp(
            -self.Etd / (self.R * self.T))) * P_0 ** 2 + self.Zfm * np.exp(-self.Efm / (self.R * self.T)) * cm * P_0 - (
                         self.F * d0) / self.V
        d1_dot = self.M_m * (self.Z_p * np.exp((-self.Ep) / (self.R * self.T)) + self.Zfm * np.exp(
            -self.Efm / (self.R * self.T))) * cm * P_0 - (self.F * di / self.V)
        return [cm_dot, ci_dot, d0_dot, d1_dot]

    def ode_step(self, state, u):
        see = solve_ivp(self.func, [0, self.dt], state, args=(u,), method='RK45', t_eval=[self.dt])
        return see.y[:, -1]
    #

    def get_trajectory(self, x_0, u_traj):
        traj = [x_0]
        for i in range(u_traj.shape[0]):
            traj.append(self.ode_step(traj[-1].reshape([-1]), u_traj[i][0]))
        return np.array(traj)
