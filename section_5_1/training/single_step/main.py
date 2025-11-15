import os
import sys
import pickle
import torch
import numpy as np

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.general.paths import PROJECT_ROOT
from utils.training_5_1_single_step import BaselineTraining

# Set seed for reproducibility
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
if __name__ == "__main__":
    ## load data
    data_path = os.path.join(PROJECT_ROOT, 'section_5_1', 'model_data', 'quadruple_tank.pickle')

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    ## Hyperparameters
    latent_space = 6
    real_states = 4
    learning_rate = 1e-3
    num_epochs = 1000
    hidden_encoder = 64
    num_training_states = 2000

    steps = [30]
    activation = ["tanh","relu"]
    optimizer = [ "Rprop","Adam"]
    ## create trainer class
    for i in steps:
        for j in activation:
            for k in optimizer:
                print(f"Training model with {i} steps, {j} activation, and {k} optimizer")
                state_space_baseline_training = BaselineTraining(latent_space, real_states, i, learning_rate, data,
                                                                 hidden_encoder, num_training_states, k, j)

                loss_arr = []
                for epoch in range(num_epochs):
                    state_space_baseline_training.optimizer.zero_grad()
                    loss = state_space_baseline_training.train_step()
                    loss.backward()
                    state_space_baseline_training.optimizer.step()
                    loss_arr.append(loss.item())
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")
                #
