import do_mpc
import numpy as np
import time

"""

Class for NMPC using do-mpc for a crystallization process. 
"""


class do_mpc_rapper_poly:
    def __init__(self, ref, ref_num, step, x_min, x_max, u_min, u_max, horizion=20, Q=np.array([1000]),
                 R=np.array([1])):
        self.t_step = 0.03  # changing according to sampling time
        self.step = step
        self.ref_num = ref_num
        self.ref = np.array([ref, ref])
        self.horizion = horizion
        self.ref = ref  # np.array([0.00045, 0.0005])
        self.Q = Q
        self.R = R
        self.xmin = x_min
        self.xmax = x_max
        self.umin = u_min
        self.umax = u_max
        self.state_scaling = np.array([1, 1e-2, 1e-3, 1e1])
        self.model = self.set_model()
        self.set_simulator()
        self.set_mpc()

    def set_mpc(self):
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': 20,
            'n_robust': 0,
            'open_loop': 0,
            't_step': self.t_step,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
            'collocation_ni': 2,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)

        _x = self.model.x
        mterm = ((_x['d1'] / _x['d0']) * self.Q[0] - self.model.tvp['temperature_external'] * self.Q[
            0]) ** 2  # terminal cost
        lterm = ((_x['d1'] / _x['d0']) * self.Q[0] - self.model.tvp['temperature_external'] * self.Q[
            0]) ** 2  # stage cost

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(F_i=self.R[0])

        for i in range(len(_x.keys())):
            self.mpc.bounds['lower', '_x', _x.keys()[i]] = self.xmin[i]
            self.mpc.bounds['upper', '_x', _x.keys()[i]] = self.xmax[i]
        _u = self.model.u
        for i in range(1, len(_u.keys())):
            self.mpc.bounds['lower', '_u', _u.keys()[i]] = self.umin[i - 1]
            self.mpc.bounds['upper', '_u', _u.keys()[i]] = self.umax[i - 1]

        for i in range(len(_x.keys())):
            self.mpc.scaling['_x', _x.keys()[i]] = self.state_scaling[i]

        self.mpc.scaling['_u', 'F_i'] = 1e-2

        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(t_now):
            for k in range(self.horizion + 1):
                current_step = int(t_now / self.t_step)
                tvp_template['_tvp', k, 'temperature_external'] = self.ref[current_step // self.step]
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()

    def set_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.model)
        params_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 0.03
        }

        self.simulator.set_param(**params_simulator)

        self.simulator.set_tvp_fun(self.simulator.get_tvp_template())
        self.simulator.setup()

    def get_mpc_trajectory(self, x_0):
        time_taken = 0
        controls = []
        states = []
        for i in range(self.step * self.ref_num):
            self.set_initial_state(x_0)

            current_time = time.time()
            u = self.mpc.make_step(x_0)
            time_taken += time.time() - current_time
            x_0 = self.simulator.make_step(u)
            # x_0 = poly_sim.get_real_state(x_0)
            controls.append(u)
            states.append(x_0)
        states = np.array(states)[:, :, 0]
        controls = np.array(controls)[:, :, 0]

        return states, controls,time_taken

    def set_initial_state(self, x_0):
        x0 = self.simulator.x0
        m = 0
        for i in x0.keys():
            x0[i] = x_0[m]
            m += 1

        self.mpc.x0 = x0
        self.simulator.x0 = x0
        self.mpc.set_initial_guess()

    def set_model(self):
        model = do_mpc.model.Model('continuous')

        Zp = 1.77 * 1e9
        Ztc = 3.8223 * 1e10
        Ep = 1.8283 * 1e4
        Ei = 1.255 * 1e5
        Etd = 2.9442 * 1e3
        Efm = 7.4478 * 1e4
        R = 8.314
        T = 335
        Etc = 2.9442 * 1e3
        V = 0.1
        C_min = 6
        C_in = 8
        F = 1
        Zfm = 1.0067 * 1e15
        Z_p = 1.77 * 1e9
        f_star = 0.58
        Zi = 3.7920 * 1e18
        Ztd = 3.1457 * 1e11
        M_m = 100.12

        C_m = model.set_variable('_x', 'C_m')
        C_i = model.set_variable('_x', 'C_i')
        d0 = model.set_variable('_x', 'd0')
        d1 = model.set_variable('_x', 'd1')

        F_i = model.set_variable('_u', 'F_i')

        P_0 = np.sqrt((2 * f_star) * C_i * Zi * np.exp((-Ei) / (R * T)) / (
                Ztd * np.exp((-Etd) / (R * T)) + Ztc * np.exp(
            (-Etc) / (T * R))))

        cm_dot = -(Zp * np.exp((-Ep / (R * T)))) * C_m * P_0 - ((F * C_m) / V) + (
                (F * C_min) / V)

        model.set_rhs('C_m', cm_dot)

        ci_dot = -Zi * np.exp((-Ei) / (R * T)) * C_i - ((F * C_i) / V) + (
                F_i * C_in) / V

        model.set_rhs('C_i', ci_dot)

        d0_dot = (0.5 * Ztc * np.exp(-Etc / (R * T)) + Ztd * np.exp(
            -Etd / (R * T))) * P_0 ** 2 + Zfm * np.exp(-Efm / (R * T)) * C_m * P_0 - (
                         F * d0) / V
        model.set_rhs('d0', d0_dot)

        d1_dot = M_m * (Z_p * np.exp((-Ep) / (R * T)) + Zfm * np.exp(
            -Efm / (R * T))) * C_m * P_0 - (F * d1 / V)

        model.set_rhs('d1', d1_dot)

        model.set_variable(var_type='_tvp', var_name='temperature_external')

        model.setup()
        return model

