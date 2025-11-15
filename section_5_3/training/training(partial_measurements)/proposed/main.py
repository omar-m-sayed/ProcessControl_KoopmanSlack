import os
import sys
import pickle

import numpy as np
import torch

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils.general.paths import PROJECT_ROOT
from utils.utils_models_for_partial_measurements.proposed_training_partial_measurements import ProposedTraining
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
        data["state"] = data["state_noise"][:, [IDX_T, IDX_L10]]
        data["label"] = data["label_noise"][:, [IDX_T, IDX_L10]]
    crystallization_sim = CrystallizationSimulate(data["state_min"], data["state_max"],
                                                  data["action_min"], data["action_max"])

    nu = 2

    # Hyperparameters
    latent_space = 10
    real_states = 2
    steps = 20
    learning_rate = 1e-3
    num_epochs = 201
    ## results for training 500 epochs
    batch_size = 2000
    hidden_encoder = 128
    hidden_decoder = 40
    num_training_states = 2000

    state_delay_vector = np.array([3])
    control_delay_vector = np.array([4])

    for i in state_delay_vector:
        for j in control_delay_vector:

            states_delay = i
            control_delay = j
            ## create trainer class
            proposed_training = ProposedTraining(latent_space, real_states, steps, learning_rate, batch_size, data,
                                                 hidden_encoder, hidden_decoder, num_training_states, states_delay,
                                                 control_delay, nu)

            ## Training Loop
            mse_arr = []
            L_10_arr = []

            for epoch in range(num_epochs):
                loss = proposed_training.train_step()
                loss.backward()

                proposed_training.opt_enc.step()
                proposed_training.opt_dec.step()

                proposed_training.opt_enc.zero_grad()
                proposed_training.opt_dec.zero_grad()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")


                if epoch % 10 == 0:
                    proposed_training.NN_decoder.eval()

                    test_input = np.array([[0.28360323, 0.6565945],
                                      [0.85812413, 0.21386546],
                                      [0.73955406, 0.66567724],
                                      [0.96417531, 0.03021013],
                                      [0.68949327, 0.48671752], ])
                    if states_delay > control_delay:
                        idx = states_delay
                    else:
                        idx = control_delay
                    input_vector = np.repeat(test_input[0, :], idx, axis=0).reshape(-1, nu)

                    inputs = np.repeat(test_input[1:, ], 20, axis=0)
                    inputs = np.vstack((input_vector, inputs))
                    states = sim_data["state"][200, :, 0]
                    label = crystallization_sim.get_trajectory(states.reshape(-1), inputs)

                    states = torch.tensor(label[:-1, :]).float()[:, [IDX_T, IDX_L10]]
                    labels = torch.tensor(label[1:, :]).float()[:, [IDX_T, IDX_L10]]
                    inputs = torch.tensor(inputs).float()

                    proposed_training.initialize_prediction_optimization()
                    if states_delay >= control_delay:
                        label = label[(states_delay + 1):, :]
                    else:
                        label = label[(control_delay + 1):, :]
                    proposed_ms = proposed_training.get_ms_prediction(states, inputs, labels).detach().numpy()
                    mse_error = np.mean((proposed_ms - label[:, [IDX_T, IDX_L10]]) ** 2)
                    proposed_training.opt = None
                    proposed_training.NN_decoder.train()

                    print("MSE Test" + str(mse_error))
                    mse_arr.append(mse_error)

            save_dir = f"proposed_noise"
            os.makedirs(save_dir, exist_ok=True)

            loss_path = os.path.join(save_dir, f"_state_delay_{states_delay}_control_delay_{control_delay}.pkl")

            with open(loss_path, 'wb') as handle:
                pickle.dump(np.asarray(mse_arr), handle, protocol=pickle.HIGHEST_PROTOCOL)

