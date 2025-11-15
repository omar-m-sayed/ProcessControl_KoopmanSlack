import torch
import numpy as np


class NARX_Dataset():
    def __init__(self, nx, nu, ndx, ndu, data_set, batch_size, training_size):

        if ndx == 0 and ndu == 0:
            self.states_delay_columns = 1
            self.states_delay_pointer = 0


            self.control_delay_columns = 0
            self.control_delay_start = 0
            self.states_delay_start = 0


        elif ndx >= ndu:
            self.states_delay_columns = ndx + 1
            self.states_delay_pointer = ndx

            self.states_delay_start = 0
            self.control_delay_columns = ndu
            self.control_delay_start = ndx - ndu

        elif ndu > ndx:
            self.states_delay_columns = ndx + 1

            self.states_delay_start = ndu - ndx
            self.states_delay_pointer = ndu
            self.control_delay_columns = ndu
            self.control_delay_start = 0





        self.states = torch.tensor(data_set["state"][:, :, 0]).float()
        self.labels = torch.tensor(data_set["label"][:, :, 0]).float()
        self.actions = torch.tensor(data_set["action"][:, :, 0]).float()

        ## states henkel matrix and vector
        self.nn_batch_states_henkel = self.get_henkel_matrix_state(batch_size)
        self.batch_states = self.states[self.states_delay_pointer:batch_size, :]
        self.nn_training_states_henkel = self.get_henkel_matrix_state(training_size)
        self.training_states = self.states[self.states_delay_pointer:training_size, :]

        ## label henkel matrix and vector
        self.nn_batch_labels_henkel = self.get_henkel_matrix_label(batch_size)
        self.batch_labels = self.labels[self.states_delay_pointer:batch_size, :]
        self.nn_training_labels_henkel = self.get_henkel_matrix_label(training_size)
        self.training_labels = self.labels[self.states_delay_pointer:training_size, :]
        ## for the control input
        self.nn_training_actions = self.actions[self.states_delay_pointer:training_size, :]
        self.nn_batch_actions = self.actions[self.states_delay_pointer:batch_size, :]

    def get_henkel_matrix_state(self, num_traj):
        hankel_matrix = torch.hstack([self.block_hankel_matrix(self.states[self.states_delay_start:num_traj, :],
                                                               self.states_delay_columns),
                                      self.block_hankel_matrix(
                                          self.actions[self.control_delay_start
                                                       :num_traj - 1, :],
                                          self.control_delay_columns)])
        return hankel_matrix

    def get_henkel_matrix_label(self, num_traj):
        hankel_matrix = torch.hstack([self.block_hankel_matrix(self.labels[self.states_delay_start:num_traj, :],
                                                               self.states_delay_columns),
                                      self.block_hankel_matrix(
                                          self.actions[self.control_delay_start + 1
                                                       :num_traj, :],
                                          self.control_delay_columns)])
        return hankel_matrix

    def get_ms_hankel(self, state, action, label):
        nn_batch_states_hankel = torch.hstack([self.block_hankel_matrix(state[self.states_delay_start:, :],
                                                                        self.states_delay_columns),
                                               self.block_hankel_matrix(
                                                   action[self.control_delay_start
                                                          : - 1, :],
                                                   self.control_delay_columns)])

        nn_batch_labels_hankel = torch.hstack([self.block_hankel_matrix(label[self.states_delay_start:, :],
                                                                        self.states_delay_columns),
                                               self.block_hankel_matrix(
                                                   action[self.control_delay_start + 1
                                                          :, :],
                                                   self.control_delay_columns)])

        action = action[self.states_delay_pointer:, :]

        return nn_batch_states_hankel, nn_batch_labels_hankel, action

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
