# from model_nvp import Encoder, Decoder
import torch
import numpy as np
import torch.nn as nn

from utils.utils_models_for_partial_measurements.inverse_opt_partial_measurements import inverseOpt
from utils.utils_models_for_partial_measurements.NARX_dataset_helper import NARX_Dataset


class ProposedTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, batch_size, data_set,
                 hidden_encoder, hidden_decoder, num_training_states, states_delay, control_delay, nu):
        """
        Initializes the model with given parameters for training and evaluation.

        Parameters:
        ----------
        latent_space : int
            Dimensionality of the latent space (e.g., the dimension of the koopman space).

        real_states : int
            Dimensionality of the real (observed) state space.

        steps : int
            Number of time steps used for multistep prediction or training horizon.

        learning_rate : float
            Learning rate for the optimizer.

        num_epochs : int
            Total number of training epochs.

        batch_size : int
            Size of the mini-batches used during training.

        data_set : Any
            Dataset object or structure used for training and evaluation (e.g., a PyTorch Dataset, tuple of arrays, etc.).
        """
        self.msModels = None
        self.c = None
        self.latent_space = latent_space
        self.steps = steps
        self.real_states = real_states
        self.nu = nu

        nu = data_set["action"].shape[1]

        self.NN_encoder = Encoder(real_states * (states_delay + 1) + nu * control_delay, hidden_encoder, latent_space,
                                  self.real_states, states_delay)
        self.NN_decoder = Decoder(real_states, hidden_decoder, latent_space)

        self.enc_params = list(self.NN_encoder.parameters())
        self.dec_params = list(self.NN_decoder.parameters())

        self.opt_enc = torch.optim.Rprop(self.enc_params, lr=1e-3)
        self.opt_dec = torch.optim.AdamW(self.dec_params, lr=1e-3,weight_decay=8e-5 )



        self.narx_data_set = NARX_Dataset(real_states, nu, states_delay, control_delay, data_set, batch_size,
                                          num_training_states)

        self.multi_step_action_training = self.block_hankel_matrix(self.narx_data_set.nn_training_actions, self.steps)
        self.multi_step_action_batch = self.block_hankel_matrix(self.narx_data_set.nn_batch_actions, self.steps)
        self.opt = None
        self.states_delay = states_delay
        self.control_delay = control_delay

    def train_step(self):
        self.c=self.inverse(self.NN_encoder.normal_forward(self.narx_data_set.nn_batch_states_henkel),
                              self.NN_decoder(self.narx_data_set.batch_states,
                                              self.NN_encoder.full_mapping(self.narx_data_set.nn_batch_states_henkel)[:,
                                              :,
                                              :self.real_states]))

        x_hat = self.NN_encoder.full_mapping(self.narx_data_set.nn_training_states_henkel)
        x_hat = self.NN_decoder.inverse(
            (self.c @ self.NN_encoder.normal_forward(self.narx_data_set.nn_training_states_henkel).T).T,
            x_hat[:, :, :self.real_states])
        #
        loss_recon = torch.mean((x_hat - self.narx_data_set.training_states) ** 2)
        #
        # ## MS_Loss
        multi_step_state = self.NN_encoder.normal_forward(self.narx_data_set.nn_batch_states_henkel)[:-(self.steps - 1),
                           :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_hankel_matrix(
            self.NN_encoder.normal_forward(self.narx_data_set.nn_batch_labels_henkel),
            self.steps)
        self.msModels =  self.inverse(combined, multi_step_label)
        #
        multi_step_state = self.NN_encoder.normal_forward(self.narx_data_set.nn_training_states_henkel)[
                           :-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_training])
        ss = (self.msModels @ combined.T).T
        ssx = self.block_hankel_matrix_3d(
            self.NN_encoder.full_mapping(self.narx_data_set.nn_training_labels_henkel)[:, :, :self.real_states],
            self.steps)
        ss_hat = self.get_ms_linear(ss, self.c, self.steps, self.latent_space, self.NN_decoder, ssx)
        ss_loss = torch.mean(((ss_hat - self.block_hankel_matrix(self.narx_data_set.training_labels, self.steps)) ** 2))
        # #
        training_labels_ss = self.block_hankel_matrix(
            self.NN_encoder.normal_forward(self.narx_data_set.nn_training_labels_henkel),
            self.steps)
        loss = loss_recon + torch.mean(
            (ss.detach() - training_labels_ss) ** 2) + ss_loss

        return loss

    @staticmethod
    def inverse(a, b):

        N, d = a.shape

        # Ensure b is (N, m)

        # Augmented system for ridge:
        # [ a          ] W ≈ [ b ]
        # [ sqrt(lam)I ]     [ 0 ]
        I = torch.eye(d, device=a.device, dtype=a.dtype)
        a_bar = torch.vstack([a, (0.2 ** 0.5) * I])
        b_bar = torch.vstack([b, torch.zeros(d, b.shape[1], device=b.device, dtype=b.dtype)])

        # Solve in least-squares sense (handles over/underdetermined and rank-deficient)
        W = torch.linalg.lstsq(a_bar, b_bar).solution  # (d, m)
        return W.T
    # def inverse(a, b):
    #     if b.ndim == 1:
    #         b = b[:, None]  # make (N, 1)
    #     N, d = a.shape
    #     I = torch.eye(d, device=a.device, dtype=a.dtype)
    #     AtA = a.T @ a
    #     Atb = a.T @ b
    #     L = torch.linalg.cholesky(AtA + 0.2 * I)             # SPD
    #     W = torch.cholesky_solve(Atb, L)                     # (d, m)
    #     return W.T

    def reconstruction(self, x):
        """
        Reconstructs the input data using the encoder and decoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, real_states).

        Returns:
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, real_states).
        """
        if self.c is None:
            raise ValueError("Model not prepared for prediction. Call prepare_for_prediction() first.")

        x_hat = self.NN_encoder.full_mapping(x)
        x_hat = self.NN_decoder.inverse((self.c @ self.NN_encoder.normal_forward(x).T).T,
                                        x_hat[:, :, :self.real_states])
        return x_hat

    @staticmethod
    def block_hankel_matrix(vector, k):
        """
        Constructs a block Hankel-like matrix from a time-series tensor.
        Parameters
        ----------
        vector : torch.Tensor
            A tensor of shape (n, d), where `n` is the number of time steps
            and `d` is the number of features per time step.
        k : int
            The number of time steps per block (window size). Must be
            less than or equal to `n`.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n - k + 1, k * d), where each row is a flattened
            version of `k` consecutive time steps from the input.

        Raises
        ------
        ValueError
            If `k` is greater than the number of available time steps `n`.

        """
        n, d = vector.shape  # Number of time steps and feature dimension
        if k > n:
            raise ValueError("k cannot be greater than the number of time steps")

        return torch.stack([vector[i:i + k].flatten() for i in range(n - k + 1)])

    @staticmethod
    def get_ms_linear(ss, c, steps, latent_space, model_decoder, sols):
        """
                Map predicted multi-step latent states back to the real state space
                in a block hankel form.

                Parameters
                ----------
                c : slack projection matrix
                ss : torch.Tensor
                    Tensor of shape (N, steps * latent_space), where each row contains
                    the latent states across multiple prediction steps.

                C : torch.Tensor
                    Linear mapping matrix of shape (n_x, latent_space) used to project
                    latent states back into the slack states.

                steps : int
                    Number of prediction steps in the trajectory.

                latent_space : int
                    Dimensionality of the latent state space.

                Returns
                -------
                torch.Tensor
                    Tensor of shape (N, steps * n_x), where each row contains the
                    reconstructed real states for all prediction steps. """

        ss_hat = 0
        for i in range(steps):
            if i == 0:
                ss_hat = model_decoder.inverse((c @ ss[:, i * latent_space:(i + 1) * latent_space].T).T,
                                               sols[:, i * latent_space:(i + 1) * latent_space])
            elif i == steps - 1:
                ss_hat = torch.hstack(
                    [ss_hat, model_decoder.inverse((c @ ss[:, -latent_space:].T).T, sols[:, -latent_space:])])
            else:
                ss_hat = torch.hstack([ss_hat,
                                       model_decoder.inverse((c @ ss[:, i * latent_space:(i + 1) * latent_space].T).T,
                                                             sols[:, i * latent_space:(i + 1) * latent_space])])

        return ss_hat

    @staticmethod
    def block_hankel_matrix_3d(vector, k):
        """
        """
        n, d, m = vector.shape
        if k > n:
            raise ValueError("k cannot be greater than the number of time steps")

        # Create sliding windows: (n - k + 1, k, d, m)
        windows = vector.unfold(0, k, 1)  # shape: (n - k + 1, k, d, m)

        # Permute to (n - k + 1, m, k, d), then reshape to (n - k + 1, k * d, m)
        windows = windows.permute(0, 2, 3, 1).reshape(n - k + 1, m, k * d)
        windows = windows.permute(0, 2, 1)  # shape: (n - k + 1, k * d, m)

        # Final output: (n - k + 1, k * d, m)
        return windows

    def initialize_prediction_optimization(self):
        """
           Initialize the inverse optimization probelm to get the real states from slack states.
           """
        self.opt = inverseOpt((self.real_states) * (self.states_delay + 1) + self.control_delay * 2, self.steps,
                              self.NN_encoder.encoder,
                              self.NN_decoder.translate_net,
                              self.NN_decoder.scale_net, self.states_delay, self.real_states)

    def get_ms_prediction(self, state, action, label):

        nn_batch_states_hankel, nn_batch_labels_hankel, action = self.narx_data_set.get_ms_hankel(state, action, label)
        if self.c is None:
            raise ValueError("Model not prepared for prediction. Call prepare_for_prediction() first.")
        if self.msModels is None:
            raise ValueError("Model not prepared for prediction. Call prepare_for_prediction() first.")

        d = nn_batch_states_hankel.shape[0]
        initial_points = np.arange(0, d, self.steps)
        initial_points = nn_batch_states_hankel[initial_points, :]
        nu = action.shape[1]

        n = int(np.ceil(d / self.steps))
        ss_states = []
        for i in range(n):
            if (i + 1) * self.steps <= d:
                x_0 = torch.hstack([self.NN_encoder.normal_forward(initial_points[i, :]),
                                    action[i * self.steps:(i + 1) * self.steps, :].flatten()])
                ss_hat = (self.msModels @ x_0).reshape(-1, self.latent_space)
                if self.opt == None:
                    ss_hat = self.NN_decoder.inverse((self.c @ ss_hat.T).T,
                                                     self.NN_encoder.full_mapping(
                                                         nn_batch_labels_hankel[i * self.steps:(i + 1) * self.steps,
                                                         :])[:, :,
                                                     :self.real_states])
                else:
                    slack_states = self.c.detach().numpy() @ ss_hat.T.detach().numpy()
                    sol = self.opt.solver(p=slack_states.T.reshape(-1), lbg=0, ubg=0)
                    opt_x_k = self.opt.opt_x(sol['x'])
                    x_opt_i = (opt_x_k['x']).T.full()[:,
                              self.real_states * self.states_delay:(self.states_delay + 1) * self.real_states]
                    ss_hat = torch.tensor(x_opt_i).float()

                ss_states.append(ss_hat)

            else:
                steps_end = d - i * self.steps
                x_0 = torch.hstack([self.NN_encoder.normal_forward(initial_points[i, :]),
                                    action[i * self.steps:(i * self.steps + steps_end), :].flatten()])
                ss_hat = (self.msModels[:self.latent_space * steps_end,
                          :self.latent_space + steps_end * nu] @ x_0).reshape(-1, self.latent_space)

                if self.opt == None:
                    ss_hat = self.NN_decoder.inverse((self.c @ ss_hat.T).T,
                                                     self.NN_encoder.full_mapping(
                                                         nn_batch_states_hankel[i * self.steps:, :])[:, :,
                                                     :self.real_states])
                    #
                    ss_states.append(ss_hat)
                else:
                    slack_states = (self.c.detach().numpy() @ ss_hat.T.detach().numpy()).T
                    slack_states=np.vstack([slack_states, np.zeros([self.steps-steps_end,slack_states.shape[1]] )])
                    sol = self.opt.solver(p=slack_states.reshape(-1), lbg=0, ubg=0)
                    opt_x_k = self.opt.opt_x(sol['x'])
                    x_opt_i = (opt_x_k['x']).T.full()[:steps_end,
                              self.real_states * self.states_delay:(self.states_delay + 1) * self.real_states]

                    ss_hat = torch.tensor(x_opt_i).float()
                    ss_states.append(ss_hat)


        return torch.vstack(ss_states)


class Encoder(nn.Module):
    """
    An encoder implemented with masking inputs.
    """

    def __init__(self, input_dim, hidden, latent_space, real_states, states_delay):
        """
        Parameters:
        ----------
        input_dim : int
            Dimension of the input space.

        hidden : int
            Number of neurons in each hidden layer.

        latent_space : int
            Dimension of the latent space representation.
        """
        super(Encoder, self).__init__()
        self.hidden = hidden
        self.latent_space = latent_space
        self.activation = nn.Tanh()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, latent_space),

        )
        self.real_states = real_states
        self.states_delay = states_delay

    def normal_forward(self, x):
        """
        Standard forward pass through the encoder.

        """
        return self.encoder(x)


    def full_mapping(self, x):
        """
        Generates latent representations for each feature masked individually and the full input.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        -------
            Tensor of shape (batch_size, latent_space, input_dim + 1) with mappings.
        """
        sol = torch.zeros([x.shape[0], self.latent_space, x.shape[1] + 1])
        for i in range(self.real_states):
            mask = torch.zeros_like(x)
            mask[:, i + self.real_states * self.states_delay] = 1
            sol[:, :, i] = self.encoder(x * mask)
        sol[:, :, x.shape[1]] = self.encoder(x)
        return sol



class Decoder(nn.Module):
    """
    Decoder that reconstructs input data from Koopman latent representations
    using learned scaling and translation networks.
    """

    def __init__(self, in_dim, hidden, latent_space):
        """
        Parameters:
        ----------
        in_dim : int
            Dimension of the output (reconstructed) data.

        hidden : int
            Number of neurons in hidden layers.

        latent_space : int
            Dimension of the latent representation.
        """
        super(Decoder, self).__init__()
        self.latent_space = latent_space
        self.hidden = hidden
        self.activation = nn.Tanh()
        self.in_dim = in_dim

        Dropout_value = 0.5

        self.scale_net = nn.Sequential(
            nn.Linear(latent_space, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Dropout(p=Dropout_value),
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Dropout(p=Dropout_value),
            nn.Linear(hidden, in_dim),

        )
        self.translate_net = nn.Sequential(
            nn.Linear(latent_space, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Dropout(p=Dropout_value),
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Dropout(p=Dropout_value),
            nn.Linear(hidden, in_dim),
        )

    def inverse(self, x, koopman):
        """
        Decodes latent representations back into the original space using learned scale and translation.
        "Notation is switched compared to the paper"

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim)

        koopman : torch.Tensor
            Koopman latent representations of shape (batch_size, latent_space, in_dim)

        Returns:
        -------
        torch.Tensor
            Reconstructed inputs.
        """
        sol = torch.zeros_like(x)
        for i in range(x.shape[1]):
            s = x * torch.exp(self.scale_net(koopman[:, :, i])) +  self.translate_net(koopman[:, :, i])
            sol[:, i] = s[:, i]
        return sol

    def forward(self, x, koopman):
        """
        Applies the inverse transformation to map from real states to slack states.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim)

        koopman : torch.Tensor
            Koopman latent representations of shape (batch_size, latent_space, in_dim)

        Returns:
        -------
        torch.Tensor
            Inverse-transformed latent input.
        """
        sol = torch.zeros_like(x)
        for i in range(x.shape[1]):
            s = (x -  self.translate_net(koopman[:, :, i])) * torch.exp(-self.scale_net(koopman[:, :, i]))
            sol[:, i] = s[:, i]
        return sol
