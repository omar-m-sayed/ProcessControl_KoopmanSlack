import os
import sys
import pickle
import torch
import numpy as np

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.general.paths import PROJECT_ROOT
from utils.training_5_1_multistep_predictors import BaselineTraining

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
    steps = 30
    learning_rate = 1e-3
    num_epochs = 1000  # results are using 1000 epochs
    batch_size = 2000
    hidden_encoder = 64
    num_training_states = 2000

    ## create trainer class activation={"tanh",relu} optimizer={"Adam", "rprop"}
    activation=  "relu"
    optimizer=   "Rprop"

    steps = [20]
    activation = ["tanh", "relu"]
    optimizer = ["Adam", "Rprop"]


    for i in steps:
        for j in activation:
            for k in optimizer:
                multistep_baseline_training = BaselineTraining(latent_space, real_states, i, learning_rate, batch_size, data,
                                                     hidden_encoder, num_training_states, k, j)
                 # training loop
                loss_arr = []
                for epoch in range(num_epochs):
                    multistep_baseline_training.optimizer.zero_grad()
                    loss = multistep_baseline_training.train_step()
                    loss.backward()
                    multistep_baseline_training.optimizer.step()
                    loss_arr.append(loss.item())
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")

                loss_arr = np.array(loss_arr)
                save_dir = f"optimizer_activation/{k}_{j}"
                os.makedirs(save_dir, exist_ok=True)

                # Save the PyTorch model
                torch.save(multistep_baseline_training,
                           os.path.join(save_dir, f"baseline_ms_{i}_steps"))

                # Save the loss array as a pickle file in the same folder
                loss_path = os.path.join(save_dir, f"baseline_ms_{i}_steps.pkl")
                with open(loss_path, 'wb') as handle:
                    pickle.dump(loss_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)



