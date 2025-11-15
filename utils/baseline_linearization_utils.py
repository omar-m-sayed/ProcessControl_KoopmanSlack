import numpy as np
from scipy.linalg import expm
import casadi.tools as ca
import scipy.sparse as sparse
import osqp
import time


class LocalLinearization:
    def __init__(self, realProcess, data, dt=0.03):
        self.realProcess = realProcess
        self.dt = dt
        self.data = data

    def run_linear_mpc_tracking(self, x_0, reference_points, Q, R, horizon=20,sim_steps=30):
        """
        Run linearized MPC using predefined reference points and track the trajectory.
        """
        steps_per_ref = sim_steps
        total_steps = steps_per_ref * len(reference_points)

        dt = self.dt
        Q = Q
        R = R

        x0_real = self.realProcess.get_real_state(x_0)[:-1].detach().numpy()
        u_min_real = self.data["action_min"]
        u_max_real = self.data["action_max"]
        x_min_real = self.data["state_min"][:-1]  ## skip last column algebric relation
        x_max_real = self.data["state_max"][:-1]

        state_traj = []
        control_traj = []
        x = x0_real
        u_prev = u_max_real
        time_taken = 0

        for i in range(total_steps):
            ref = reference_points[i // steps_per_ref, :]

            ## Linearization  also added to the time taken

            current_time = time.time()
            A = self.realProcess.Poly.A_fun(x, u_prev)
            B = self.realProcess.Poly.B_fun(x, u_prev)
            A_d = self.compute_discrete_a_matrix(A, dt)
            B_d = self.compute_discrete_b_matrix(A, B, dt)
            ctrl = linearized_controller(
                A_d, B_d, horizon,
                umin=u_min_real, umax=u_max_real,
                xmin=x_min_real * 0, xmax=x_max_real,
                Q=Q, R=np.eye(1) * 10000, nx=A.shape[0], nu=B.shape[1], C=np.eye(A.shape[0]), su=A.shape[0],
            )
            ## control weight used for stability reasons since the states are in there orignal magnitudes
            ## removing this weight should give the same result



            # llmpc = Linear_MPC(A_d, B_d, 20, Q, np.eye(1) * 0.000, x_min_real, x_max_real, u_min_real,
            #                    u_max_real)
            ctrl.set_reference(ref)
            _, _, x_next, u = ctrl.step(x)
            time_taken += time.time() - current_time

            state_traj.append(x)
            control_traj.append(u[0])
            x = self.realProcess.Poly.ode_step(x, u[0])
            u_prev = u

        return np.array(state_traj), np.array(control_traj),time_taken

    def get_ms_linearized(self, state, action):
        state = self.realProcess.get_real_state(state)
        x_0 = state[0, :]  ## initial state
        x_0_real = self.realProcess.get_real_state(x_0)[:4]
        action_real = self.realProcess.get_real_action(action)
        u_prev = action[0, :]
        arr = []
        for i in range(action.shape[0]):
            A = self.realProcess.Poly.A_fun(x_0_real, u_prev)
            B = self.realProcess.Poly.B_fun(x_0_real, u_prev)
            A_d = self.compute_discrete_a_matrix(A, self.dt)
            B_d = self.compute_discrete_b_matrix(A, B, self.dt)
            x = A_d @ x_0_real + B_d @ u_prev
            x_0_real = x.full().reshape([-1])

            if i % 20 == 0:
                x = state[i, :4].reshape([-1, 1])
                x_0_real = x

            arr.append(x)

            u_prev = action_real[i, :]

        arr = np.array(arr)[:, :, 0]
        # NAMW = arr[:, 3] / arr[:, 2]
        # arr = np.hstack([arr, NAMW.reshape(-1, 1)])
        return arr

    @staticmethod
    def compute_discrete_b_matrix(A_continuous: np.ndarray, B_continuous: np.ndarray, dt: float) -> np.ndarray:
        """
        Discretizes the continuous-time input matrix B using the exact integral of the matrix exponential.

        B_d = ∫₀^dt e^(Aτ) B dτ = A⁻¹ (e^{A·dt} - I) B

        Args:
            A_continuous (np.ndarray): Continuous-time system matrix (n x n).
            B_continuous (np.ndarray): Continuous-time input matrix (n x m).
            dt (float): Time step for discretization.

        Returns:
            np.ndarray: Discrete-time input matrix B_d (n x m).
        """
        A = A_continuous
        B = B_continuous
        n = A.shape[0]
        I = np.eye(n)

        return np.linalg.pinv(A) @ (expm(A * dt) - I) @ B

    @staticmethod
    def compute_discrete_a_matrix(A_continuous: np.ndarray, dt: float) -> np.ndarray:
        """
        Discretizes the continuous-time system matrix A using the matrix exponential.

        Args:
            A_continuous (np.ndarray): Continuous-time system matrix (n x n).
            dt (float): Time step for discretization.

        Returns:
            np.ndarray: Discrete-time system matrix A_d (n x n).
        """
        return expm(A_continuous * dt)


class linearized_controller:
    def __init__(self, A, B, horizon, umin, umax, xmin, xmax, Q, R, nx, nu, C=np.array(None), isMulti_step=False,
                 Qn=None, Exception=False, su=5, remove_initial_state_constraint=True):
        self.su = su

        self.x_struct = ca.struct_symMX([
            ca.entry("x", shape=(self.su,), repeat=[horizon + 1]),
            ca.entry("u", shape=(nu,), repeat=[horizon + 1]),
        ])
        self.AA = A
        self.B = B
        self.N = horizon

        self.Q = Q

        if Qn is None:
            self.QN = self.Q * 0
        else:
            self.QN = sparse.diags(Qn) * 0

        self.R = R
        self.C = C

        self.nu = nu
        self.nx = nx
        # initialize initial point and x_r NEEDS TO BE CHANGED
        x0 = np.zeros([A.shape[0]])
        u0 = np.zeros(nu)

        diff_matrix = sparse.diags([1, -1], [0, 1], shape=(self.N + 1, self.N + 1))
        delta_u_matrix = sparse.kron(diff_matrix.T @ diff_matrix, R).tocsc()
        xr = np.zeros(self.su)
        self.min = xmin
        self.max = xmax
        # - quadratic objective
        self.P = sparse.block_diag(
            [np.eye(self.nx) * 0, sparse.kron(sparse.eye(self.N), self.Q), self.QN,
             delta_u_matrix], format='csc')
        # - linear objective
        self.q = np.hstack(
            [np.ones(self.nx) * 0, np.kron(np.ones(self.N), -self.Q @ xr), -self.QN @ xr,
             np.zeros((self.N + 1) * self.nu)])

        # - linear dynamics
        if isMulti_step:
            Ax = -np.eye(self.nx + self.su * (self.N + 1))
            Ax[self.nx:self.nx + self.su, :self.nx] = C

            for i in range(1, self.N + 1):
                Ax[(self.nx + self.su + self.su * (i - 1)):self.su * (i) + self.nx + self.su, :self.nx] = C @ A[:, (
                                                                                                                           i - 1) * self.nx:self.nx * i]

            Ax = sparse.bsr_matrix(Ax)
            # Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), B)
        else:
            A_ms = self.A_from_single_to_multi(A, self.N)
            B_ms = self.B_from_single_to_multi(A, B, self.N)

            Ax = -np.eye(self.nx + self.su * (self.N + 1))
            Ax[self.nx:self.nx + self.su, :self.nx] = C

            for i in range(1, self.N + 1):
                Ax[(self.nx + self.su + self.su * (i - 1)):self.su * (i) + self.nx + self.su, :self.nx] = C @ A_ms[:, (
                                                                                                                              i - 1) * self.nx:self.nx * i]

            Ax = sparse.bsr_matrix(Ax)
            B = B_ms

        B_matrix = self.create_B_matrix(self.nx, self.N, B)
        B_matrix = np.hstack([np.zeros((B_matrix.shape[0], nu)), B_matrix])

        Bu = sparse.bsr_matrix(B_matrix)
        Aeq = sparse.hstack([Ax, Bu])
        self.leq = np.hstack([-x0, np.zeros((self.N + 1) * self.su)])
        self.ueq = self.leq
        # - input and state constraints
        # if C.any() == None:

        Aineq = sparse.block_diag(
            [sparse.eye(self.nx), sparse.kron(sparse.eye(self.N + 1), np.eye(self.su)),
             sparse.eye((self.N + 1) * self.nu)])

        U_0 = np.array([2, 1])
        # Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)

        if remove_initial_state_constraint:
            lineq = np.hstack(
                [np.ones(self.nx) * -np.inf, np.ones(self.su) * -np.inf, np.kron(np.ones(self.N), xmin),
                 np.kron(np.ones(self.N + 1), umin)])
            uineq = np.hstack(
                [np.ones(self.nx) * +np.inf, np.ones(self.su) * np.inf, np.kron(np.ones(self.N), xmax),
                 np.kron(np.ones(self.N + 1), umax)])
        else:
            lineq = np.hstack(
                [np.ones(self.nx) * -np.inf, np.kron(np.ones(self.N + 1), xmin), np.kron(np.ones(self.N + 1), umin)])
            uineq = np.hstack(
                [np.ones(self.nx) * +np.inf, np.kron(np.ones(self.N + 1), xmax), np.kron(np.ones(self.N + 1), umax)])
        # - OSQP constraints
        self.A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([self.leq, lineq])
        self.u = np.hstack([self.ueq, uineq])
        self.xmin = xmin
        self.xmax = xmax
        self.umax = umax
        self.umin = umin

        # Setup OSQRP solver
        self.prob = osqp.OSQP()
        self.set_tolerence(1e-5, 9e-4)


    def set_tolerence(self, eps_abs, eps_rel):
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, eps_abs=eps_abs, eps_rel=eps_rel)

    def B_from_single_to_multi(self, A, B, N):
        B_matrix = np.zeros([B.shape[0], B.shape[1] * N])
        for i in range(0, N):
            B_matrix[:, (i) * B.shape[1]:(i + 1) * B.shape[1]] = np.linalg.matrix_power(A, i) @ B

        return B_matrix

    def A_from_single_to_multi(self, A, N):
        A_matrix = np.zeros([A.shape[0], A.shape[0] * N])
        for i in range(1, N + 1):
            A_matrix[:, (i - 1) * A.shape[0]:(i) * A.shape[0]] = np.linalg.matrix_power(A, i)

        return A_matrix

    def custom_flip(self, B, nu):
        B_flip = np.zeros_like(B)
        end = B.shape[1] // nu
        for i in range(B.shape[1] // nu):
            B_flip[:, i * nu:(i + 1) * nu] = B[:, (end - i - 1) * nu:(end - i) * nu]
        return B_flip

    def create_B_matrix(self, nx, N, B):
        matrix = np.zeros((nx + self.su * (N + 1), N * self.nu))
        for i in range(1, N + 1):
            matrix[nx + self.su + self.su * (i - 1):self.su * (i) + nx + self.su,
            :i * self.nu] = self.C @ self.custom_flip(
                B[:, :i * self.nu], self.nu)
        return matrix

    def solve_receding_horizon(self, x_0, n_sim):
        optimal_state = [x_0.reshape([-1])]
        optimal_control = []
        for i in range(n_sim):
            _, _, x_0, u = self.step(optimal_state[-1])
            optimal_state.append(x_0.reshape([-1]))
            optimal_control.append(u)

        return np.asarray(optimal_state), np.asarray(optimal_control)

    def solve_receding_horizon_star(self, x_0, n_sim):
        optimal_state = [x_0.reshape([-1])]
        optimal_control = []
        for i in range(n_sim):
            _, _, x_0, u = self.step(optimal_state[-1])
            optimal_state.append(x_0.reshape([-1]))
            optimal_control.append(u)

        return np.asarray(optimal_state), np.asarray(optimal_control)

    def set_reference(self, x_r):
        self.q = np.hstack(
            [np.zeros(self.nx), np.kron(np.ones(self.N), -self.Q @ x_r), -self.QN @ x_r,
             np.zeros((self.N + 1) * self.nu)])
        self.prob.update(q=self.q)

    def set_initial_point(self, x_0):
        self.l[:self.nx] = -x_0
        self.u[:self.nx] = -x_0
        self.prob.update(l=self.l, u=self.u)

    def set_plant_model(self):
        pass

    def set_input_initial_point(self, u0):
        lineq = np.hstack(
            [np.ones(self.nx) * -np.inf, np.kron(np.ones(self.N + 1), self.xmin), u0,
             np.kron(np.ones(self.N), self.umin)])
        uineq = np.hstack(
            [np.ones(self.nx) * +np.inf, np.kron(np.ones(self.N + 1), self.xmax), u0,
             np.kron(np.ones(self.N), self.umax)])
        self.l = np.hstack([self.leq, lineq])
        self.u = np.hstack([self.ueq, uineq])
        self.prob.update(l=self.l, u=self.u)

    def step(self, x_0, u_0=0, set_control=False):
        if set_control:
            self.set_input_initial_point(u_0)
        self.set_initial_point(x_0)
        res = self.prob.solve()

        # # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        u = res.x[-self.N * self.nu:-(self.N - 1) * self.nu]
        sol = self.x_struct(res.x[self.nx:])
        u_0 = np.array(sol["u", :])[0, :, 0]
        return np.array(sol["x", :])[:, :, 0], np.array(sol["u", :])[1:, :, 0], sol["x", 1].full(), u
