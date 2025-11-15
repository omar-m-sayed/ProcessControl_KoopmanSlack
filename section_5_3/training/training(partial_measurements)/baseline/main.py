import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils.general.paths import PROJECT_ROOT
from utils.utils_models_for_partial_measurements.baseline_training_partial_measurements import BaselineTraining
from section_5_3.model_data.model_crystallization import CrystallizationSimulate

# Set seed for reproducibility
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

IDX_T = 1  # column for temperature T
IDX_L10 = 6  # column for L10

if __name__ == "__main__":
    ## load data
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'model_data', 'Crystillization_noise.pickle')
    with open(data_path, "rb") as handle:
        sim_data = pickle.load(handle)
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
        data["state"] = data["state_noise"][:, [IDX_T, IDX_L10], :]
        data["label"] = data["label_noise"][:, [IDX_T, IDX_L10], :]
    crystallization_sim = CrystallizationSimulate(data["state_min"], data["state_max"],
                                                  data["action_min"], data["action_max"])

    ## Hyperparameters
    latent_space = 10
    real_states = 2  # includes the relation L10
    steps = 20
    learning_rate = 1e-3
    num_epochs = 201
    batch_size = 2000  # used for identifying linear models
    hidden_encoder = 128
    num_training_states = 2000  # used for training the NN
    nu = 2
    states_delay = 3
    control_delay = 4
    ## create trainer class
    baseline_training = BaselineTraining(latent_space, real_states, nu, steps, learning_rate, batch_size, data,
                                         hidden_encoder, num_training_states, states_delay, control_delay)

    # training loop
    state_delay_vector = np.array([3])
    control_delay_vector = np.array([1])

    for i in state_delay_vector:
        for j in control_delay_vector:

            states_delay = i
            control_delay = j
            ## create trainer class
            baseline_training = BaselineTraining(latent_space, real_states, nu, steps, learning_rate, batch_size, data,
                                                 hidden_encoder, num_training_states, states_delay, control_delay)

            ## Training Loop
            mse_arr = []
            L_10_arr = []

            for epoch in range(num_epochs):
                baseline_training.optimizer.zero_grad()
                loss = baseline_training.train_step()
                loss.backward()
                baseline_training.optimizer.step()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")


                if (epoch) % 10 == 0:

                    input = np.array([[0.28360323, 0.6565945],
                                      [0.85812413, 0.21386546],
                                      [0.73955406, 0.66567724],
                                      [0.96417531, 0.03021013],
                                      [0.68949327, 0.48671752], ])
                    if states_delay > control_delay:
                        idx = states_delay
                    else:
                        idx = control_delay
                    input_vector = np.repeat(input[0, :], idx, axis=0).reshape(-1, nu)

                    inputs = np.repeat(input[1:, ], 20, axis=0)
                    inputs = np.vstack((input_vector, inputs))
                    states = sim_data["state"][200, :, 0]
                    label = crystallization_sim.get_trajectory(states.reshape(-1), inputs)

                    states = torch.tensor(label[:-1, :]).float()[:, [IDX_T, IDX_L10]]
                    labels = torch.tensor(label[1:, :]).float()[:, [IDX_T, IDX_L10]]
                    inputs = torch.tensor(inputs).float()

                    if states_delay >= control_delay:
                        label = label[(states_delay + 1):, :]
                    else:
                        label = label[(control_delay + 1):, :]
                    proposed_ms = baseline_training.get_ms_prediction(states, inputs).detach().numpy()
                    mse_error = np.mean((proposed_ms - label[:, [IDX_T, IDX_L10]]) ** 2)


                    print("MSE Test" + str(mse_error))
                    mse_arr.append(mse_error)

            save_dir = f"baselineNoise"
            os.makedirs(save_dir, exist_ok=True)

            # Save the PyTorch model

            # Save the loss array as a pickle file in the same folder
            loss_path = os.path.join(save_dir, f"_state_delay_{states_delay}_control_delay_{control_delay}.pkl")

            with open(loss_path, 'wb') as handle:
                pickle.dump(np.asarray(mse_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)
