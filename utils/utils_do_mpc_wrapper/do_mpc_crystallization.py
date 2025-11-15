import time
import do_mpc
import numpy as np

"""

Class for NMPC using do-mpc for a crystallization process. 

"""


class do_mpc_rapper_crystalization:
    def __init__(self, ref, ref_num, step, x_min, x_max, u_min, u_max, horizion=20, Q=np.array([1000]),
                 R=np.array([10, 10])):
        self.t_step = 5  # changing according to sampling time
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
        self.state_scaling = np.array([1, 1e2, 1e2, 1e8, 1e4, 1e2])
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
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': True,
        }

        self.mpc.set_param(**setup_mpc)

        _x = self.model.x
        mterm = ((_x['u_1'] / _x['u_0']) * self.Q[0] - self.model.tvp['temperature_external'] * self.Q[
            0]) ** 2  # terminal cost
        lterm = ((_x['u_1'] / _x['u_0']) * self.Q[0] - self.model.tvp['temperature_external'] * self.Q[
            0]) ** 2  # stage cost

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(F_J=self.R[0])
        self.mpc.set_rterm(F_Feed=self.R[1])  # penalty on input changes

        for i in range(len(_x.keys())):
            self.mpc.bounds['lower', '_x', _x.keys()[i]] = self.xmin[i]
            self.mpc.bounds['upper', '_x', _x.keys()[i]] = self.xmax[i]
        _u = self.model.u
        for i in range(1, len(_u.keys())):
            self.mpc.bounds['lower', '_u', _u.keys()[i]] = self.umin[i - 1]
            self.mpc.bounds['upper', '_u', _u.keys()[i]] = self.umax[i - 1]

        for i in range(len(_x.keys())):
            self.mpc.scaling['_x', _x.keys()[i]] = self.state_scaling[i]

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
            't_step': 5
        }

        self.simulator.set_param(**params_simulator)

        self.simulator.set_tvp_fun(self.simulator.get_tvp_template())
        self.simulator.setup()

    def get_mpc_trajectory(self, x_0):
        total_time = 0

        controls = []
        states = []

        for i in range(self.step * self.ref_num):
            self.set_initial_state(x_0)

            current_time = time.time()
            u = self.mpc.make_step(x_0)
            total_time += time.time() - current_time
            x_0 = self.simulator.make_step(u)
            controls.append(u)
            states.append(x_0)
        states = np.array(states)[:, :, 0]
        controls = np.array(controls)[:, :, 0]

        return states, controls, total_time

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

        rho = 1043
        cp = 4
        T_feed = 350
        C_feed = 0.11238 * np.exp(9.0849e-3 * (T_feed - 273.15))
        A = 10
        U = 1000
        cpj = 4
        V = 10

        m = rho * V
        rhoj = 1050
        mu_0_in = 1.0292 * 1e8
        mu_1_in = 4.117 * 1e4
        mu_2_in = 1.7501 * 1e1
        rho_cryst = 1432
        delta_H_cyst = 0
        k_v_cryst = np.pi / 6
        VJ = 10
        m_j = rhoj * VJ

        C = model.set_variable('_x', 'C')
        T = model.set_variable('_x', 'T')
        T_J = model.set_variable('_x', 'T_J')
        u_0 = model.set_variable('_x', 'u_0')
        u_1 = model.set_variable('_x', 'u_1')
        u_2 = model.set_variable('_x', 'u_2')

        c_star = 0.11238 * np.exp(9.0849e-3 * (T - 273.15))
        delta_s = (C - c_star) / c_star
        G = 5.857 * 1e-5 * delta_s ** 2 * np.tanh(0.9113 / delta_s)
        m_dot_cryst = 3 * V * k_v_cryst * rho_cryst * G * u_2

        c_star = 0.11238 * np.exp(9.0849e-3 * (T - 273.15))

        F_J = model.set_variable('_u', 'F_J')
        F_Feed = model.set_variable('_u', 'F_Feed')

        dot_C = (1 / m) * (-m_dot_cryst + rho * F_Feed * (C_feed - C))

        model.set_rhs('C', dot_C)

        dot_T = (1 / (m * cp)) * (-delta_H_cyst * m_dot_cryst + rho * F_Feed * cp * (
                T_feed - T) - U * A * (T - T_J))

        model.set_rhs('T', dot_T)
        T_j_in = 300

        dot_t_j = (1 / (m_j * cpj)) * (
                rhoj * F_J * cp * (T_j_in - T_J) - U * A * (T_J - T))

        model.set_rhs('T_J', dot_t_j)

        d_mu_0 = ((rho * F_Feed) / m) * (mu_0_in - u_0) + (C - c_star) * 1e5

        model.set_rhs('u_0', d_mu_0)

        d_mu_1 = G * u_0 + ((rho * F_Feed) / m) * (mu_1_in - u_1)

        model.set_rhs('u_1', d_mu_1)

        d_mu_2 = 2 * G * u_1 + ((rho * F_Feed) / m) * (mu_2_in - u_2)

        model.set_rhs('u_2', d_mu_2)
        self.temperature_external = model.set_variable(var_type='_tvp', var_name='temperature_external')

        model.setup()
        return model
