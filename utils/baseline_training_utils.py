# from model_nvp import Encoder, Decoder
import torch
import numpy as np
import torch.nn as nn


class BaselineTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, batch_size, data_set,
                 hidden_encoder, num_training_states):
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

        self.NN_encoder = Encoder_base_line(real_states, hidden_encoder, latent_space)

        params = (
            list(self.NN_encoder.parameters())
        )
        self.optimizer = torch.optim.Rprop(params, lr=learning_rate)
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
        multi_step_state = self.NN_encoder(self.nn_batch_states)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_henkel_matrix(self.NN_encoder(self.nn_batch_labels), self.steps)
        self.msModels = self.inverse(combined, multi_step_label)
        multi_step_state = self.NN_encoder(self.nn_training_states)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_training])
        ss = (self.msModels @ combined.T).T
        ss_hat = self.get_ms_linear(ss, self.c, self.steps, self.latent_space)
        ss_loss = torch.mean(((ss_hat - self.block_henkel_matrix(self.nn_training_labels, self.steps)) ** 2))
        #
        training_labels_ss = self.block_henkel_matrix(self.NN_encoder(self.nn_training_labels), self.steps)
        loss = loss_recon + ss_loss + torch.mean((ss - training_labels_ss) ** 2)

        return loss

    @staticmethod
    def inverse(a, b):
        """
         Solve a regularized least-squares problem to approximate the mapping
         between input matrix `a` and target matrix `b`.
          Parameters
    ----------
        a : torch.Tensor
            Input matrix of shape (N, d)

        b : torch.Tensor
            Target matrix of shape (N, m)

        Returns
        -------
        torch.Tensor
            Solution matrix of shape (m, d)
             """

        N, d = a.shape
        I = torch.eye(d, device=a.device, dtype=a.dtype)
        a_bar = torch.vstack([a, (0.2 ** 0.5) * I])
        b_bar = torch.vstack([b, torch.zeros(d, b.shape[1], device=b.device, dtype=b.dtype)])
        W = torch.linalg.lstsq(a_bar, b_bar).solution
        return W.T

    def prepare_for_prediction(self):
        self.c = self.inverse(self.NN_encoder(self.nn_batch_states), self.nn_batch_states)
        multi_step_state = self.NN_encoder(self.nn_batch_states)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_henkel_matrix(self.NN_encoder(self.nn_batch_labels), self.steps)
        self.msModels = self.inverse(combined, multi_step_label)

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

        """
        n, d = vector.shape  # Number of time steps and feature dimension
        if k > n:
            raise ValueError("k cannot be greater than the number of time steps")

        return torch.stack([vector[i:i + k].flatten() for i in range(n - k + 1)])

    @staticmethod
    def get_ms_linear(ss, C, steps, latent_space):
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
                latent states back into the real state space.

            steps : int
                Number of prediction steps in the trajectory.

            latent_space : int
                Dimensionality of the latent state space.

            Returns
            -------
            torch.Tensor
                Tensor of shape (N, steps * n_x), where each row contains the
                reconstructed real states for all prediction steps. """

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
        Perform multi-step state prediction given states and action sequences.

        This method propagates the system forward in time using the learned
        multi-step model (`self.msModels`). It reconstructs predicted states
        either through the linear matrix C

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
            Predicted state trajectory of shape (d, n_x), reconstructed by the C matrix. """

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


class Encoder_base_line(nn.Module, ):
    def __init__(self, in_num, first_hidden, latent_space):
        """
           Baseline feedforward encoder network for mapping real states into
           a higher-dimensional latent space.
           Parameters
           ----------
           in_num : int
               Dimensionality of the input (real state space).

           first_hidden : int
               Number of neurons in each hidden layer. The same value is used
               for all three hidden layers.

           latent_space : int
               Dimensionality of the output latent space.


           """

        super().__init__()
        self.latent_space = latent_space
        self.relu_activation = nn.Tanh()

        self.encoder = nn.Sequential(
            nn.Linear(in_num, first_hidden),
            self.relu_activation,
            nn.Linear(first_hidden, first_hidden),
            self.relu_activation,
            nn.Linear(first_hidden, first_hidden),
            self.relu_activation,
            nn.Linear(first_hidden, latent_space),
        )

    def forward(self, input):
        return self.encoder(input)
