import torch
import numpy as np
import torch.nn as nn


class BaselineTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, batch_size, data_set,
                 hidden_encoder, num_training_states, optimizer, activation):
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

        self.NN_encoder = Encoder_base_line(real_states, hidden_encoder, latent_space, activation)

        params = (
            list(self.NN_encoder.parameters())
        )
        if optimizer == "Rprop":
            self.optimizer = torch.optim.Rprop(params, lr=learning_rate)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not implemented")

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

    def train_step(self):
        ## Reconstruction loss
        self.c = self.inverse(self.NN_encoder(self.nn_batch_states), self.nn_batch_states)
        x_hat = self.NN_encoder(self.nn_training_states)
        x_hat = (self.c @ x_hat.T).T
        loss_recon = torch.mean((x_hat - self.nn_training_states) ** 2)

        ## MS_Loss
        if self.steps > 1:
            multi_step_state = self.NN_encoder(self.nn_batch_states)[:-(self.steps - 1), :]
        else:
            multi_step_state = self.NN_encoder(self.nn_batch_states)[:, :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_henkel_matrix(self.NN_encoder(self.nn_batch_labels), self.steps)
        self.msModels = self.inverse(combined, multi_step_label)

        if self.steps > 1:
            multi_step_state = self.NN_encoder(self.nn_training_states)[:-(self.steps - 1), :]
        else:
            multi_step_state = self.NN_encoder(self.nn_training_states)[:, :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_training])
        ss = (self.msModels @ combined.T).T
        ss_hat = self.get_ms_linear(ss, self.c, self.steps, self.latent_space)

        ss_loss = torch.mean(((ss_hat - self.block_henkel_matrix(self.nn_training_labels, self.steps)) ** 2))
        #
        training_labels_ss = self.block_henkel_matrix(self.NN_encoder(self.nn_training_labels),
                                                      self.steps)
        loss = loss_recon + ss_loss + torch.mean((ss - training_labels_ss) ** 2)

        return loss

    @staticmethod
    def inverse(a, b):
        """
        Solves the linear system A @ X = B for X using QR decomposition.

        This avoids explicitly computing A⁻¹ and improves numerical stability.

        Parameters
        ----------
        a : torch.Tensor
            A square matrix of shape (n, n) to invert.

        b : torch.Tensor
            Right-hand side matrix or vector of shape (n, m) or (n,).

        Returns
        -------
        torch.Tensor
            Solution X of shape (m, n) such that A @ X.T = B.
        """
        q, r = torch.linalg.qr(a)
        return torch.linalg.solve_triangular(r, q.T @ b, upper=True).T

    # def prepare_for_prediction(self):
    #     self.c = self.inverse(self.NN_encoder(self.nn_batch_states), self.nn_batch_states)
    #     if self.steps > 1:
    #         multi_step_state = self.NN_encoder(self.nn_batch_states)[:-(self.steps - 1), :]
    #     else:
    #         multi_step_state = self.NN_encoder(self.nn_batch_states)[:, :]
    #     combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
    #     multi_step_label = self.block_henkel_matrix(self.NN_encoder(self.nn_batch_labels), self.steps)
    #     self.msModels = self.inverse(combined, multi_step_label)

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

        x_hat = (self.c @ self.NN_encoder(x).T).T

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
    def get_ms_linear(ss, C, steps, latent_space):
        """
         Applies a linear mapping C to each latent block of a multi-step Hankel Matrix.

         Parameters
         ----------
         ss : torch.Tensor
             A tensor of shape (n_samples, steps * latent_space), where each row
             contains a sequence of latent states concatenated across the prediction horizon.

         C : torch.Tensor
             Linear mapping matrix of shape (output_dim, latent_space) used to map
             latent states back to the real state space (or any desired output space).

         steps : int
             The number of prediction steps (blocks) in the horizon.

         latent_space : int
             Dimensionality of the latent state space.

         Returns
         -------
         torch.Tensor
             A tensor of shape (n_samples, steps * output_dim), where each block of size
             `output_dim` corresponds to the mapped real state for one step of the horizon.

         Notes
         -----
          This function returns a block Hankel matrix of the estimated real states, corresponding
          to the multi-step predictions in the latent space.
         """
        for i in range(steps):
            if i == 0:
                ss_hat = (C @ ss[:, i * latent_space:(i + 1) * latent_space].T).T
            elif i == steps - 1:
                ss_hat = torch.hstack([ss_hat, (C @ ss[:, -latent_space:].T).T])
            else:
                ss_hat = torch.hstack([ss_hat, (C @ ss[:, i * latent_space:(i + 1) * latent_space].T).T])

        return ss_hat

    def get_ms_prediction(self, state, action):
        """
          Perform multi-step prediction in the original state space using the
          learned linear multi-step model.

          Parameters
          ----------
          state : torch.Tensor
              Tensor of shape (T, state_dim), the sequence of observed states
              over time. This is used to reinitialize the initial state after k-steps

          action : torch.Tensor
              Tensor of shape (T, action_dim), the sequence of control inputs
              aligned with `state`.

          Returns
          -------
          torch.Tensor
              Predicted states of shape

        """

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
                x_0 = torch.hstack([self.NN_encoder(initial_points[i, :]),
                                    action[i * self.steps:(i + 1) * self.steps, :].flatten()])
                ss_hat = (self.msModels @ x_0).reshape(-1, self.latent_space)
                ss_hat = (self.c @ ss_hat.T).T

                ss_states.append(ss_hat)

            else:
                steps_end = d - i * self.steps
                x_0 = torch.hstack([self.NN_encoder(initial_points[i, :]),
                                    action[i * self.steps:(i * self.steps + steps_end), :].flatten()])
                ss_hat = (self.msModels[:self.latent_space * steps_end,
                          :self.latent_space + steps_end * nu] @ x_0).reshape(-1, self.latent_space)
                ss_hat = (self.c @ ss_hat.T).T

                ss_states.append(ss_hat)

        return torch.vstack(ss_states)

class Encoder_base_line(nn.Module, ):
    def __init__(self, in_num, first_hidden, latent_space,activation):
        """
        Baseline feed-forward encoder network for mapping inputs into a latent space.

        This encoder consists of two hidden layers with a configurable activation
        function and an output layer projecting into the latent space.

        Parameters
        ----------
        in_num : int
            Dimensionality of the input features.

        first_hidden : int
            Number of units in each hidden layer.

        latent_space : int
            Dimensionality of the latent space (output of the encoder).

        activation : str
            Activation function to use in hidden layers. Must be either:
            - "relu": Rectified Linear Unit
            - "tanh": Hyperbolic tangent
        """


        super().__init__()
        self.latent_space = latent_space

        if activation == "relu":
            self.relu_activation = nn.ReLU()
        elif activation == "tanh":
            self.relu_activation = nn.Tanh()
        else:
            raise ValueError(f"Activation function {activation} not implemented. Use 'relu' or 'tanh'.")

        self.encoder = nn.Sequential(
            nn.Linear(in_num, first_hidden),
            self.relu_activation,
            nn.Linear(first_hidden, first_hidden),
            self.relu_activation,
            # nn.Linear(first_hidden, first_hidden),
            # self.relu_activation,
            nn.Linear(first_hidden, latent_space),
        )

    def forward(self, input):
        return self.encoder(input)
