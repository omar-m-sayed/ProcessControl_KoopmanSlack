import torch
import numpy as np
import scipy.sparse as sparse
import osqp
import time

class BaselineControl:
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model,nx,eps_abs=1e-2, eps_rel=1e-2):

        self.model_encoder = model_encoder
        self.real_model = real_model
        self.latent_space = latent_space
        self.nx = nx


        self.controller = LinearControllerOSQPProposed(A.detach().numpy(), B.detach().numpy(), steps, u_min, u_max,
                                                       x_min,
                                                       x_max, Q, R, latent_space, nu, C=C.detach().numpy(), nx=nx)

        # using lower tolerence will lead to OSQP not finding a solution due to constraint violations
        self.controller.set_tolerance(eps_abs, eps_rel)

    def closed_loop_control(self, x_0_0, reference_points, reference_num, ref_idx, num_steps):
        if x_0_0.dtype != torch.float32:
            x_0_0 = torch.tensor(x_0_0).float()

        state_trajectory_arr = []

        x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()
        state_trajectory_arr.append(x_0.reshape([1, -1]))

        plant_trajectory_arr = [x_0_0.detach().numpy()]
        input_trajectory_arr = []
        ref_trajectory_arr = []
        ref = np.zeros(5)

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                x_ref = ref
                self.controller.set_reference(x_ref.reshape([-1]))
            _, _, next_state, next_input, time_taken = self.controller.step(x_0)

            state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_0.detach().numpy().reshape([-1]), next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, :].reshape([1, -1])).float()
            x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()
            ref_trajectory_arr.append(ref)

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return state_trajectory_arr, plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr


class BaselineControlNarx(BaselineControl):
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model, nx, states_delay, control_delay, eps_abs=1e-2, eps_rel=1e-2):

        super().__init__(A, B, C, model_encoder, steps, latent_space, x_max,
                    x_min, u_max, u_min, Q, R, nu, real_model, nx, eps_abs, eps_rel)
        ## using lower tolerence will lead to OSQP not finding a solution due to constraint violations
        self.controller.set_tolerance(eps_abs, eps_rel)
        self.states_delay = states_delay
        self.control_delay = control_delay

    def closed_loop_control(self, x_0_real, state_vector, input_vector, reference_points, reference_num, ref_idx,
                            num_steps):


        if state_vector.dtype != torch.float32:
            state_vector = torch.tensor(state_vector).float()
        if input_vector.dtype != torch.float32:
            input_vector = torch.tensor(input_vector).float()

        state_trajectory_arr = []
        initial_point_ = torch.hstack((state_vector.reshape(-1), input_vector.reshape(-1)))

        x_0 = self.model_encoder(initial_point_.reshape([1, -1])).detach().numpy()
        state_trajectory_arr.append(x_0.reshape([1, -1]))

        plant_trajectory_arr = [x_0_real]
        input_trajectory_arr = []
        ref_trajectory_arr = []

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                x_ref = ref
                self.controller.set_reference(x_ref.reshape([-1]))
            _, _, next_state, next_input, time_taken = self.controller.step(x_0)

            state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_real.reshape([-1]),
                                                   next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, [1, 6]]).float()
            state_vector = torch.vstack([state_vector, x_0_0])
            input_vector = torch.vstack([input_vector, torch.tensor(next_input).float()])
            x_0_0 = torch.hstack((state_vector[-(self.states_delay + 1):, :].reshape(-1),
                                  input_vector[-self.control_delay:, :].reshape(-1)))
            x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()
            ref_trajectory_arr.append(ref)
            x_0_real = plant[1, :]

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr


class LinearControllerOSQPProposed:
    def __init__(self, A, B, horizon, u_min, u_max, x_min, x_max, Q, R, nz, nu, C, nx):
        self.nx = nx
        self.nu = nu
        self.nz = nz

        self.Q = Q
        self.QN = self.Q

        self.horizon = horizon
        self.N = horizon
        x0 = np.zeros([A.shape[0]])
        xr = np.zeros(self.nx)
        # - quadratic objective
        self.P = sparse.block_diag(
            [np.eye(self.nz) * 0, sparse.kron(sparse.eye(self.N), self.Q), self.QN,
             sparse.kron(sparse.eye(self.N + 1), R)], format='csc')
        # - linear objective
        self.q = np.hstack(
            [np.ones(self.nz) * 0, np.kron(np.ones(self.N), -self.Q @ xr), -self.QN @ xr,
             np.zeros((self.N + 1) * self.nu)])

        # - linear dynamics
        Ax = -np.eye(self.nz + self.nx * (self.N + 1))
        Ax[self.nz:self.nz + self.nx, :self.nz] = C
        for i in range(1, self.N + 1):
            Ax[(self.nz + self.nx + self.nx * (i - 1)):self.nx * i + self.nz + self.nx, :self.nz] = C @ A[:, (
                                                                                                                     i - 1) * self.nz:self.nz * i]
        Ax = sparse.bsr_matrix(Ax)

        B_matrix = self.create_b_matrix(self.nz, self.N, B, C)
        B_matrix = np.hstack([np.zeros((B_matrix.shape[0], nu)), B_matrix])

        Bu = sparse.bsr_matrix(B_matrix)
        Aeq = sparse.hstack([Ax, Bu])
        self.leq = np.hstack([-x0, np.zeros((self.N + 1) * self.nx)])
        self.ueq = self.leq

        A_ineq = sparse.block_diag(
            [sparse.eye(self.nz), sparse.kron(sparse.eye(self.N + 1), np.eye(self.nx)),
             sparse.eye((self.N + 1) * self.nu)])

        l_ineq = np.hstack(
            [np.ones(self.nz) * -np.inf, np.kron(np.ones(self.N + 1), x_min), np.kron(np.ones(self.N + 1), u_min)])
        u_ineq = np.hstack(
            [np.ones(self.nz) * +np.inf, np.kron(np.ones(self.N + 1), x_max), np.kron(np.ones(self.N + 1), u_max)])

        self.A = sparse.vstack([Aeq, A_ineq], format='csc')
        self.l = np.hstack([self.leq, l_ineq])
        self.u = np.hstack([self.ueq, u_ineq])

        # Setup OSQP solver
        self.prob = osqp.OSQP()

    def create_b_matrix(self, nz, N, B, C):
        matrix = np.zeros((nz + self.nx * (N + 1), N * self.nu))

        b_new = np.zeros([C.shape[0] * N, self.nu * N])
        for i in range(N):
            for j in range(N):
                b_new[i * C.shape[0]:(i + 1) * C.shape[0], j * self.nu:(j + 1) * self.nu] = (C @
                                                                                             B[i * self.nz:(
                                                                                                                   i + 1) * self.nz,
                                                                                             j * self.nu:(
                                                                                                                 j + 1) * self.nu])

        matrix[nz + self.nx:, :] = b_new

        return matrix

    def set_tolerance(self, eps_abs, eps_rel):
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, )

    def set_reference(self, x_r):
        self.q = np.hstack(
            [np.zeros(self.nz), np.kron(np.ones(self.N), -self.Q @ x_r), -self.QN @ x_r,
             np.zeros((self.N + 1) * self.nu)])
        # self.q = np.hstack([self.q, self.add_one])

        self.prob.update(q=self.q)

    def set_initial_point(self, x_0):
        self.l[:self.nz] = -x_0
        self.u[:self.nz] = -x_0
        self.prob.update(l=self.l, u=self.u)

    def step(self, x_0):
        current_time = time.time()
        self.set_initial_point(x_0)
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        current_time = time.time() - current_time

        states = res.x[self.nz:self.nz + self.nx * (self.horizon + 1)].reshape([-1, self.nx])
        control = res.x[self.nz + self.nx * (self.horizon + 1):].reshape([-1, self.nu])

        return states, control[1:, :], states[1, :], control[1, :], current_time
