import torch
import numpy as np
import scipy.sparse as sparse
import osqp
import time


class proposedControl:
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model, model_decoder=None, nx=5, eps_abs=1e-3, eps_rel=1e-3):
        """
        Initialize a closed-loop MPC controller with encoder/decoder models.

        This constructor prepares state and input constraints, transforms
        bounds through the encoder/decoder pair, and instantiates an
        OSQP-based linear MPC controller.

        Parameters
        ----------
        A : torch.Tensor
            System dynamics matrix

        B : torch.Tensor
            Input dynamics matrix

        C : torch.Tensor
            Linear mapping for the Koopman states to the slack states

        model_encoder : torch.nn.Module
              mapping real states to koopman states

        steps : int
            Prediction horizon (number of steps).

        latent_space : int
            Dimension of the Koopman space.

        x_max : np.ndarray
            Upper bounds on the real state

        x_min : np.ndarray
            Lower bounds on the real state

        u_max : np.ndarray
            Upper bounds on the control input

        u_min : np.ndarray
            Lower bounds on the control input

        Q : np.ndarray
            State cost matrix.

        R : np.ndarray
            Input cost matrix.

        nu : int
            Dimension of the control input.

        real_model : object
            Plant model providing `get_trajectory` for simulating real dynamics.

        model_decoder : torch.nn.Module, optional
        mapping slack states back to real state space.

        nx : int, default=5
            Dimension of the real state space.

        eps_abs : float, default=1e-2
            Absolute solver tolerance for OSQP.

        eps_rel : float, default=1e-2
            Relative solver tolerance for OSQP. """

        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.real_model = real_model
        self.latent_space = latent_space
        self.nx = nx

        # compute constraints
        x_min_dec, x_max_dec = self.get_state_constraints(x_max, x_min)

        self.controller = LinearControllerOSQPProposed(A.detach().numpy(), B.detach().numpy(), steps, u_min, u_max,
                                                       x_min_dec, x_max_dec, Q,
                                                       R, latent_space, nu, C=C.detach().numpy(), nx=nx)

        self.controller.set_tolerance(eps_abs, eps_rel)

    def get_state_constraints(self, x_max, x_min):
        x_max_t = torch.tensor(x_max.reshape([1, -1])).float()
        x_min_t = torch.tensor(x_min.reshape([1, -1])).float()

        x_max_dec = self.model_decoder(x_max_t, self.model_encoder.full_mapping(x_max_t)).detach().numpy().reshape(-1)
        x_min_dec = self.model_decoder(x_min_t, self.model_encoder.full_mapping(x_min_t)).detach().numpy().reshape(-1)

        # Ensure bounds have correct order
        return np.minimum(x_min_dec, x_max_dec), np.maximum(x_min_dec, x_max_dec)


    def closed_loop_control(self, x_0_0, reference_points, reference_num, ref_idx, num_steps):
        """
            Run closed-loop control simulation with the learned model and MPC controller.

            This method simulates the closed-loop system for a given number of
            reference points and steps, applying the controller at each time
            step and propagating the plant dynamics.

            Parameters
            ----------
            x_0_0 : torch.Tensor or np.ndarray
                Initial state of the plant. If not in `torch.float32` format,
                it is converted automatically.

            reference_points : list or np.ndarray
                Sequence of reference values for the controlled state dimension
                specified by `ref_idx`. Length should equal `reference_num`.

            reference_num : int
                Number of reference segments. Each reference point is held
                constant for `num_steps` simulation steps.

            ref_idx : int
                Index of the state dimension to which the reference trajectory
                is applied.

            num_steps : int
                Number of time steps to simulate per reference point.

            Returns
            -------
            state_trajectory_arr : np.ndarray
                Sequence of predicted states over the simulation horizon,
                aligned with plant trajectory.

            plant_trajectory_arr : np.ndarray
                Sequence of actual plant states simulated from the real model.

            input_trajectory_arr : np.ndarray
                Sequence of applied control inputs over the horizon.

            ref_trajectory_arr : np.ndarray
                Sequence of reference points

            total_time_taken : float
                Accumulated solver time over the entire simulation. """

        total_time_taken = 0

        if x_0_0.dtype != torch.float32:
            x_0_0 = torch.tensor(x_0_0).float()

        state_trajectory_arr = []
        xms_arr = []

        x_0 = self.model_encoder.normal_forward(x_0_0.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
        state_trajectory_arr.append(
            (self.model_decoder(torch.tensor(x_0_0).float().reshape([1, -1]), self.model_encoder.full_mapping(
                torch.tensor(x_0_0).float().reshape([1, -1])))).reshape([1, -1]).detach().numpy())
        xms_arr.append(x_0)

        plant_trajectory_arr = [x_0_0.detach().numpy()]
        input_trajectory_arr = []
        ref_trajectory_arr = []
        ref = np.zeros(self.nx)

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                ref = torch.tensor(ref).float().reshape([1, -1])
                x_ref = self.model_decoder(ref, self.model_encoder.full_mapping(ref)).detach().numpy()
                self.controller.set_reference(x_ref.reshape([-1]))

            _, _, next_state, next_input, time_taken = self.controller.step(x_0)
            total_time_taken += time_taken

            state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_0.detach().numpy().reshape([-1]), next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, :].reshape([1, -1])).float()
            x_0 = self.model_encoder.normal_forward(x_0_0.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
            ref_trajectory_arr.append(ref)

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return state_trajectory_arr, plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr, total_time_taken


class proposedControlNarx(proposedControl):
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model, model_decoder=None, nx=5, states_delay=0, control_delay=0,
                 eps_abs=1e-5, eps_rel=1e-5):
        """NARX version of the MPC controller (adds delay handling)."""
        self.states_delay = states_delay
        self.control_delay = control_delay
        self.nu = nu
        super().__init__(A, B, C, model_encoder, steps, latent_space, x_max,
                         x_min, u_max, u_min, Q, R, nu, real_model,
                         model_decoder=model_decoder, nx=nx,
                         eps_abs=eps_abs, eps_rel=eps_rel)



    def get_state_constraints(self, x_max, x_min):
        x_max_t = torch.tensor(x_max.reshape([1, -1])).float()
        x_min_t = torch.tensor(x_min.reshape([1, -1])).float()

        x_max_dec = self.model_decoder(x_max_t, self.model_encoder.full_mapping(
            self.batch_zeros(x_max_t, True).reshape([1, -1]))).detach().numpy().reshape(-1)
        x_min_dec = self.model_decoder(x_min_t, self.model_encoder.full_mapping(
            self.batch_zeros(x_min_t, True).reshape([1, -1]))).detach().numpy().reshape(-1)

        # Ensure bounds have correct order
        return np.minimum(x_min_dec, x_max_dec), np.maximum(x_min_dec, x_max_dec)

    def batch_zeros(self, state, return_torch=False):
        if return_torch:
            batched = torch.zeros(((self.states_delay + 1) * self.nx + self.control_delay * self.nu,))
            batched[self.nx * self.states_delay:self.nx * (self.states_delay + 1)] = state

        else:
            batched = np.zeros(((self.states_delay + 1) * self.nx + self.control_delay * self.nu,))
            batched[self.nx * self.states_delay:self.nx * (self.states_delay + 1)] = state

        return batched

    def closed_loop_control(self, x_0_real, state_vector, input_vector, reference_points, reference_num, ref_idx,
                            num_steps):
        """
            Run closed-loop control simulation with the learned model and MPC controller.

            This method simulates the closed-loop system for a given number of
            reference points and steps, applying the controller at each time
            step and propagating the plant dynamics.

            Parameters
            ----------
            x_0_0 : torch.Tensor or np.ndarray
                Initial state of the plant. If not in `torch.float32` format,
                it is converted automatically.

            reference_points : list or np.ndarray
                Sequence of reference values for the controlled state dimension
                specified by `ref_idx`. Length should equal `reference_num`.

            reference_num : int
                Number of reference segments. Each reference point is held
                constant for `num_steps` simulation steps.

            ref_idx : int
                Index of the state dimension to which the reference trajectory
                is applied.

            num_steps : int
                Number of time steps to simulate per reference point.

            Returns
            -------
            state_trajectory_arr : np.ndarray
                Sequence of predicted states over the simulation horizon,
                aligned with plant trajectory.

            plant_trajectory_arr : np.ndarray
                Sequence of actual plant states simulated from the real model.

            input_trajectory_arr : np.ndarray
                Sequence of applied control inputs over the horizon.

            ref_trajectory_arr : np.ndarray
                Sequence of reference points

            total_time_taken : float
                Accumulated solver time over the entire simulation. """

        total_time_taken = 0

        # if x_0_0.dtype != torch.float32:
        #     x_0_0 = torch.tensor(x_0_0).float()

        if state_vector.dtype != torch.float32:
            state_vector = torch.tensor(state_vector).float()
        if input_vector.dtype != torch.float32:
            input_vector = torch.tensor(input_vector).float()

        # state_trajectory_arr = []
        initial_point_ = torch.hstack((state_vector.reshape(-1), input_vector.reshape(-1)))
        x_0 = self.model_encoder.normal_forward(initial_point_.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
        # state_trajectory_arr.append(
        #     (self.model_decoder(torch.tensor(x_0_0).float().reshape([1, -1]), self.model_encoder.full_mapping(
        #         torch.tensor(x_0_0).float().reshape([1, -1])))).reshape([1, -1]).detach().numpy())

        plant_trajectory_arr = [x_0_real]
        input_trajectory_arr = []
        ref_trajectory_arr = []

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                ref_batch = self.batch_zeros(ref)

                ref = torch.tensor(ref).float().reshape([1, -1])
                ref_batch = torch.tensor(ref_batch).float().reshape([1, -1])

                x_ref = self.model_decoder(ref, self.model_encoder.full_mapping(ref_batch)).detach().numpy()
                self.controller.set_reference(x_ref.reshape([-1]))

            _, _, next_state, next_input, time_taken = self.controller.step(x_0)
            total_time_taken += time_taken

            # state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_real, next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, [1, 6]]).float()
            state_vector = torch.vstack([state_vector, x_0_0])
            input_vector = torch.vstack([input_vector, torch.tensor(next_input).float()])
            x_0_0 = torch.hstack((state_vector[-(self.states_delay + 1):, :].reshape(-1),
                                  input_vector[-self.control_delay:, :].reshape(-1)))
            x_0 = self.model_encoder.normal_forward(x_0_0.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
            ref_trajectory_arr.append(ref)
            x_0_real = plant[1, :]

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr, total_time_taken


class LinearControllerOSQPProposed:
    def __init__(self, A, B, horizon, u_min, u_max, x_min, x_max, Q, R, nz, nu, C, nx):
        """
        Initialize quadratic MPC problem in OSQP (block formulation).

        This builds a convex QP of the form
            minimize    (1/2) zᵀ P z + qᵀ z
            subject to  A z ∈ [l, u]

        Inputs
        ------
        A : np.ndarray
            Multi-step latent transition matrix
         B : np.ndarray
            Input transition per step matrix
        horizon : int
            Prediction horizon length N.
        u_min, u_max : np.ndarray (shape: (nu,))
            Input box bounds applied to each stage k = 0…N.
        x_min, x_max : np.ndarray (shape: (nx,))
            State box bounds applied to each stage k = 0…N.
        Q : np.ndarray (shape: (nx, nx))
            Stage state tracking penalty.
        R : np.ndarray (shape: (nu, nu))
            Stage input penalty.
        nz : int
            Size of the koopman space.
        nu : int
            Input dimension.
        C : np.ndarray (shape: (nx, nz))
            Linear map  from koopman states to slack satates.
        nx : int
            Slack state dimension. """
        self.nx = nx
        self.nu = nu
        self.nz = nz

        self.Q = Q
        self.QN = self.Q * 100

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
        """

        :param nz: size of koopman states
        :param N: horizon
        :param B: multi-step tau
        :param C: linear map from koopman states to slack states
        :return: A matrix C*B where be is multiplied by each block cell
        """
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
        """
        Configure numerical tolerances for the OSQP solver.

        Parameters
        ----------
        eps_abs : float
            Absolute tolerance
        eps_rel : float
            Relative tolerance
            This method needs to be called after the initialization of the class
            and before calling the step method
            """
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, )

    def set_reference(self, x_r):
        """
         to set the reference point
         """
        self.q = np.hstack(
            [np.zeros(self.nz), np.kron(np.ones(self.N), -self.Q @ x_r), -self.QN @ x_r,
             np.zeros((self.N + 1) * self.nu)])
        self.prob.update(q=self.q)

    def set_initial_point(self, x_0):
        """
        This method is used to set the initial point of the optimization problem
        :param x_0:  intial point
        :return: set the initial point
        """

        self.l[:self.nz] = -x_0
        self.u[:self.nz] = -x_0
        self.prob.update(l=self.l, u=self.u)

    def step(self, x_0):
        """
        Method used to calculate the optimal control action. set initial_point
        and reference needs  to be called before this method

        :param x_0: intial point
        :return:  return optimal parameters
        """
        current_time = time.time()
        self.set_initial_point(x_0)
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        current_time = time.time() - current_time

        states = res.x[self.nz:self.nz + self.nx * (self.horizon + 1)].reshape([-1, self.nx])
        control = res.x[self.nz + self.nx * (self.horizon + 1):].reshape([-1, self.nu])

        return states, control[1:, :], states[1, :], control[1, :], current_time
