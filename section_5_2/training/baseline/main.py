import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.general.paths import PROJECT_ROOT

from utils.baseline_training_utils import BaselineTraining
import torch
import numpy as np


# Set seed for reproducibility

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
if __name__ == "__main__":
    ## load data

    data_path = os.path.join(PROJECT_ROOT, 'section_5_2', 'model_data', 'polymerization_dataset.pickle')
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    ## Hyperparameters
    latent_space = 10
    real_states = 5  # includes the relation NAMW
    steps = 20
    learning_rate = 1e-3
    num_epochs = 200  # results are using 200 epochs
    batch_size = 5000  # used for identifying linear models
    hidden_encoder = 128
    num_training_states = 20000  # results are 20000 used for training the NN

    ## create trainer class
    baseline_training = BaselineTraining(latent_space, real_states, steps, learning_rate, batch_size, data,
                                         hidden_encoder, num_training_states)

    # training loop
    for epoch in range(num_epochs):
        baseline_training.optimizer.zero_grad()
        loss = baseline_training.train_step()
        loss.backward()
        baseline_training.optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")

