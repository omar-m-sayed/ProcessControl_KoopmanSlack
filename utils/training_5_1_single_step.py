import torch
import numpy as np
import torch.nn as nn


class BaselineTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, data_set,
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
        data_set : np. Array
            Dataset object or structure used for training
        """
        self.msModels = None
        self.c = None
        self.latent_space = latent_space
        self.steps = steps
        self.real_states = real_states
        self.NN_encoder = Encoder_base_line(real_states, hidden_encoder, latent_space, activation)
        self.c = nn.parameter.Parameter(torch.ones([real_states, latent_space], requires_grad=True) * 0.5)
        self.system_matrix = [nn.parameter.Parameter(
            torch.hstack([torch.eye(latent_space, requires_grad=True) * 0.5, torch.zeros(latent_space, 2)]))]

        params = (
                list(self.NN_encoder.parameters()) + list(self.system_matrix) + list([self.c]))

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
        self.real_states = real_states
        self.multi_step_action_training = self.block_henkel_matrix(self.nn_training_actions, self.steps)

    def train_step(self):
        ## Reconstruction loss
        if self.steps > 1:
            x_hat = (self.c @ (self.NN_encoder(self.nn_training_states)).T).T
        else:
            self.c = torch.linalg.lstsq(
                self.NN_encoder(self.nn_training_states),
                self.nn_training_states
            ).solution.T
            x_hat = (self.c @ (self.NN_encoder(self.nn_training_states)).T).T
        loss_recon = torch.mean((x_hat - self.nn_training_states) ** 2)


        ## MS_loss
        label_henkel = BaselineTraining.block_henkel_matrix(self.NN_encoder(self.nn_training_labels), self.steps)


        hat_henkel = self.multi_step_single(self.system_matrix[0][:, :self.latent_space],
                                            self.system_matrix[0][:, self.latent_space:], self.steps,
                                            self.multi_step_action_training,
                                            self.NN_encoder(self.nn_training_states)[:, :])


        ss_hat = self.get_ms_linear(hat_henkel.T, self.c, self.steps, self.latent_space)
        ss_loss = torch.mean((ss_hat - BaselineTraining.block_henkel_matrix(self.nn_training_labels, self.steps)) ** 2)
        ss_loss_en = torch.mean((hat_henkel.T - label_henkel) ** 2)
        loss = loss_recon + ss_loss + ss_loss_en
        return loss

    @staticmethod
    def multi_step_single(a_matrix, b_matrix, steps, action, states):
        if steps > 1:
            states = states[:-(steps - 1), :].T
        else:
            states = states.T
        states_hat = []
        b_columns = b_matrix.shape[1]
        for i in range(steps):
            if i == 0:
                states_hat.append(
                    (a_matrix @ states).T + (b_matrix @ action[:, i * b_columns:(i + 1) * b_columns].T).T)
            else:
                states_hat.append(
                    (a_matrix @ states_hat[-1].T).T + (b_matrix @ action[:, i * b_columns:(i + 1) * b_columns].T).T)

        return torch.hstack(states_hat).T

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
        ss_hat = []
        for i in range(steps):
            if i == 0:
                ss_hat = (C @ ss[:, i * latent_space:(i + 1) * latent_space].T).T
            elif i == steps - 1:
                ss_hat = torch.hstack([ss_hat, (C @ ss[:, -latent_space:].T).T])
            else:
                ss_hat = torch.hstack([ss_hat, (C @ ss[:, i * latent_space:(i + 1) * latent_space].T).T])

        return ss_hat

    def multi_step_prediction(self, x_data, action):
        """
        Perform multi-step state prediction using the identified linear system matrices.

        Parameters
        ----------
        x_data : torch.Tensor
            Input states of shape (T, state_dim), where T is the number of time steps.
            These are first mapped into the latent space using the encoder.

        action : torch.Tensor
            Control inputs of shape (T, action_dim), aligned with `x_data` over time.

        Returns
        -------
        torch.Tensor
            Predicted states of shape (T, real_state_dim), obtained by propagating
            through the linear model in the latent space and mapping back to the
            original state space with matrix `c`.

        Notes
        -----
        - The latent dynamics are modeled as:
            z_{k+1} = A z_k + B u_k
          where `A` and `B` are extracted from `self.system_matrix`.

        - At the start of each horizon segment (when `i % self.steps == 0`),
          the true encoded state `x_data[i, :]` is used to initialize the prediction.
          Otherwise, the previously predicted state is propagated forward.

        - After simulating in the latent space, the predicted latent sequence is
          mapped back into the real state space via `(self.c @ z.T).T`.
        """

        x_data = self.NN_encoder(x_data)
        a_matrix = self.system_matrix[0][:, :self.latent_space]
        b_matrix = self.system_matrix[0][:, self.latent_space:]
        x_hat = []
        for i in range(action.shape[0]):
            if i % self.steps == 0 :
                x_0 = x_data[i, :]
                x_hat.append(a_matrix @ x_0 + b_matrix @ action[i, :])

            else:
                x_hat.append(a_matrix @ x_hat[-1] + b_matrix @ action[i, :])

        x_hat = torch.stack(x_hat)
        x_hat = (self.c@x_hat.T).T
        return x_hat


class Encoder_base_line(nn.Module, ):
    def __init__(self, in_num, first_hidden, latent_space, activation):
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
            nn.Linear(first_hidden, latent_space),
        )

    def forward(self, input):
        return self.encoder(input)
