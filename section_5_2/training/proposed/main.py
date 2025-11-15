import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.general.paths import PROJECT_ROOT
from utils.proposed_training_utils import ProposedTraining
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


    # Hyperparameters
    latent_space = 10
    real_states = 5
    steps = 20
    learning_rate = 1e-3
    num_epochs = 200 ## results is for training 200 epochs
    batch_size = 5000
    hidden_encoder = 128
    hidden_decoder = 32
    num_traing_states = 20000

    ## create trainer class
    proposed_training = ProposedTraining(latent_space, real_states, steps, learning_rate, batch_size, data,
                                         hidden_encoder, hidden_decoder, num_traing_states)

    for epoch in range(num_epochs):
        loss = proposed_training.train_step()
        loss.backward()

        proposed_training.opt_enc.step()
        proposed_training.opt_dec.step()

        proposed_training.opt_enc.zero_grad()
        proposed_training.opt_dec.zero_grad()
        #  proposed_training.optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")



