# from model_nvp import Encoder, Decoder
import torch
import numpy as np
import torch.nn as nn
from utils.general.inverse_opt import inverseOpt


class ProposedTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, batch_size, data_set,
                 hidden_encoder, hidden_decoder, num_training_states):
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

        self.NN_encoder = Encoder(real_states, hidden_encoder, latent_space)
        self.NN_decoder = Decoder(real_states, hidden_decoder, latent_space)

        enc_params = list(self.NN_encoder.parameters())
        dec_params = list(self.NN_decoder.parameters())

        self.opt_enc = torch.optim.Rprop(enc_params, lr=1e-3)  # encoder
        self.opt_dec = torch.optim.Adam(dec_params, lr=1e-3, weight_decay=1e-3)
        ## the dimension is a set of states only used during training
        self.nn_training_states = torch.tensor(data_set["state"][:num_training_states, :, 0]).float()
        self.nn_training_labels = torch.tensor(data_set["label"][:num_training_states, :, 0]).float()
        self.nn_training_actions = torch.tensor(data_set["action"][:num_training_states, :, 0]).float()
        ## states used for identifying MS models
        self.nn_batch_states = torch.tensor(data_set["state"][:batch_size, :, 0]).float()
        self.nn_batch_labels = torch.tensor(data_set["label"][:batch_size, :, 0]).float()
        self.nn_batch_actions = torch.tensor(data_set["action"][:batch_size, :, 0]).float()

        self.real_states = real_states
        test_index = np.random.randint(0, data_set["state"].shape[2])
        self.nn_testing_states = torch.tensor(data_set["state"][test_index:test_index + 100, :, 1]).float()
        self.multi_step_action_training = self.block_henkel_matrix(self.nn_training_actions, self.steps)
        self.multi_step_action_batch = self.block_henkel_matrix(self.nn_batch_actions, self.steps)
        self.opt = None

    def train_step(self):
        ## Reconstruction loss
        self.c = self.inverse(self.NN_encoder.normal_forward(self.nn_batch_states),
                              self.NN_decoder(self.nn_batch_states,
                                              self.NN_encoder.full_mapping(self.nn_batch_states)[:, :,
                                              :self.real_states]))

        x_hat = self.NN_encoder.full_mapping(self.nn_training_states)
        x_hat = self.NN_decoder.inverse((self.c @ self.NN_encoder.normal_forward(self.nn_training_states).T).T,
                                        x_hat[:, :, :self.real_states])

        loss_recon = torch.mean((x_hat - self.nn_training_states) ** 2)

        ## MS_Loss
        multi_step_state = self.NN_encoder.normal_forward(self.nn_batch_states)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_henkel_matrix(self.NN_encoder.normal_forward(self.nn_batch_labels), self.steps)
        self.msModels = self.inverse(combined, multi_step_label)

        multi_step_state = self.NN_encoder.normal_forward(self.nn_training_states)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_training])
        ss = (self.msModels @ combined.T).T
        ssx = self.block_hankel_matrix_3d(
            self.NN_encoder.full_mapping(self.nn_training_labels)[:, :, :self.real_states], self.steps)
        ss_hat = self.get_ms_linear(ss, self.c, self.steps, self.latent_space, self.NN_decoder, ssx)
        ss_loss = torch.mean(((ss_hat - self.block_henkel_matrix(self.nn_training_labels, self.steps)) ** 2))

        training_labels_ss = self.block_henkel_matrix(self.NN_encoder.normal_forward(self.nn_training_labels),
                                                      self.steps)
        loss = loss_recon + ss_loss + torch.mean(
            (ss - training_labels_ss) ** 2)

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
    def block_henkel_matrix(vector, k):
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
                in a block henkel form.

                Parameters
                ----------
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
        Constructs a block Hankel-like matrix for each 2D matrix along the 3rd dimension of a 3D tensor.

        Parameters
        ----------
        vector : torch.Tensor
            A tensor of shape (n, d, m), where `n` is the number of time steps,
            `d` is the number of features, and `m` is the number of samples/slices.

        k : int
            The number of time steps per block (window size). Must be <= n.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n - k + 1, k * d, m), where each slice [:, :, i]
            is the Hankel matrix from vector[:, :, i].
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
        self.opt = inverseOpt(self.real_states, self.steps, self.NN_encoder.encoder,
                              self.NN_decoder.translate_net,
                              self.NN_decoder.scale_net)

    def get_ms_prediction(self, state, action):
        """
         Perform multi-step state prediction given states and action sequences.

         This method propagates the system forward in time using the learned
         multi-step model (`self.msModels`). It reconstructs predicted states
         either through the decoder network or, if an optimization problem
         (`self.opt`) is provided, by solving for feasible states that satisfy
         the inverse transformation constraints.

         Parameters
         ----------
         state : torch.Tensor
             Tensor of shape (d, n_x) containing the sequence of input states,
             where d is the trajectory length and n_x is the dimension of
             the real state space.

         action : torch.Tensor
             Tensor of shape (d, n_u) containing the sequence of control inputs,
             where n_u is the input dimension.

         Returns
         -------
         torch.Tensor
             Predicted state trajectory of shape (d, n_x), reconstructed either
             via the decoder or via optimization if `self.opt` is defined. """
        if self.c is None:
            raise ValueError("Model not prepared for prediction. Call prepare_for_prediction() first.")
        if self.msModels is None:
            raise ValueError("Model not prepared for prediction. Call prepare_for_prediction() first.")

        d = state.shape[0]
        initial_points = np.arange(0, d, self.steps)
        initial_points = state[initial_points, :]
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
                                                         state[i * self.steps:(i + 1) * self.steps, :])[:, :,
                                                     :self.real_states])
                else:
                    slack_states=self.c.detach().numpy() @ ss_hat.T.detach().numpy()
                    sol=self.opt.solver(p=slack_states.T.reshape(-1), lbg=0, ubg=0)
                    opt_x_k = self.opt.opt_x(sol['x'])
                    x_opt_i = (opt_x_k['x']).T.full()[:, :]
                    ss_hat = torch.tensor(x_opt_i).float()

                ss_states.append(ss_hat)

            else:
                steps_end = d - i * self.steps
                x_0 = torch.hstack([self.NN_encoder.normal_forward(initial_points[i, :]),
                                    action[i * self.steps:(i * self.steps + steps_end), :].flatten()])
                ss_hat = (self.msModels[:self.latent_space * steps_end,
                          :self.latent_space + steps_end * nu] @ x_0).reshape(-1, self.latent_space)
                ss_hat = self.NN_decoder.inverse((self.c @ ss_hat.T).T,
                                                     self.NN_encoder.full_mapping(
                                                         state[i * self.steps:, :])[:, :,
                                                     :self.real_states])

                ss_states.append(ss_hat)

        return torch.vstack(ss_states)


class Encoder(nn.Module):
    """
    A more advanced encoder that allows for analysis of individual feature contributions
    and supports full or partial input masking.
    """

    def __init__(self, input_dim, hidden, latent_space):
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

    def normal_forward(self, x):
        """
        Standard forward pass through the encoder.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        -------
        torch.Tensor
            Encoded latent representation.
        """
        return self.encoder(x)

    def forward(self, x):
        """
        Forward pass where each input feature is masked individually and then summed with the full input.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        -------
        torch.Tensor
            Aggregated latent representation based on masked inputs.
        """
        sol = 0
        for i in range(x.shape[1]):
            mask = torch.zeros_like(x)
            mask[:, i] = 1
            if i == 0:
                sol = self.encoder(x * mask)
            else:
                sol += self.encoder(x * mask)
        sol += self.encoder(x)
        return sol

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
        for i in range(x.shape[1]):
            mask = torch.zeros_like(x)
            mask[:, i] = 1
            sol[:, :, i] = self.encoder(x * mask)
        sol[:, :, x.shape[1]] = self.encoder(x)
        return sol

    def mapping(self, full_state, state_index):
        """
        Generates the latent representation of a single input feature masked.

        Parameters:
        ----------
        full_state : torch.Tensor
            Full input state tensor.

        state_index : int
            Index of the feature to isolate.

        Returns:
        -------
        torch.Tensor
            Latent representation of masked input.
        """
        x = torch.zeros_like(full_state)
        x[:, state_index] = full_state[:, state_index]
        mask = torch.zeros_like(full_state)
        mask[:, state_index] = 1
        return self.encoder(full_state * mask)


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

        self.scale_net = nn.Sequential(
            nn.Linear(latent_space, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, in_dim),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(latent_space, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, in_dim),
        )

    def inverse(self, x, koopman):
        """
        Decodes latent representations back into the original space using learned scale and translation.

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
            s = x * torch.exp(self.scale_net(koopman[:, :, i])) + self.translate_net(koopman[:, :, i])
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
            s = (x - self.translate_net(koopman[:, :, i])) * torch.exp(-self.scale_net(koopman[:, :, i]))
            sol[:, i] = s[:, i]
        return sol
