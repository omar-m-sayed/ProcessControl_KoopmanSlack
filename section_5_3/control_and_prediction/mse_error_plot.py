import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.general.paths import PROJECT_ROOT
from utils.general.plot_helper import set_plot_style
from section_5_3.model_data.model_crystallization import CrystallizationSimulate


seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

FIG_WIDTH_IN = 3.448

if __name__ == "__main__":

    steps = 20
    num_real_states = 7


    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'model_data', 'crystallization_data.pickle')
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    ## load the models
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'control_and_prediction', 'learned_models')

    proposed_model = torch.load(os.path.join(data_path, "proposed_crystallization_model"), weights_only=False)
    baseline_model = torch.load(os.path.join(data_path, "baseline_crystallization_model"), weights_only=False)
    crystallization_sim = CrystallizationSimulate(data["state_min"], data["state_max"],
                                                  data["action_min"], data["action_max"])

    input_step = np.random.randint(0, 100, [100, 2]) / 100
    inputs = np.repeat(input_step, 20, axis=0)
    states = data["state"][0, :, 0]

    label = crystallization_sim.get_trajectory(states.reshape(-1), inputs)

    states = torch.tensor(label[:-1, :]).float()
    inputs = torch.tensor(inputs[:, :]).float()
    label = label[1:, :]

    proposed_model.initialize_prediction_optimization()
    proposed_ms = proposed_model.get_ms_prediction(states, inputs).detach().numpy()
    baseline_ms = baseline_model.get_ms_prediction(states, inputs).detach().numpy()

    label_ = np.reshape(label, (steps, -1, num_real_states), order="F")
    label_ = label_.transpose(0, 2, 1)

    baseline_ms_ = baseline_ms.reshape(steps, -1, num_real_states, order="F")
    baseline_ms_ = baseline_ms_.transpose(0, 2, 1)
    ms_model_ = proposed_ms.reshape(steps, -1, num_real_states, order="F")
    ms_model_ = ms_model_.transpose(0, 2, 1)

    baseline_ms_ = np.mean((baseline_ms_ - label_) ** 2, axis=2)
    baseline_ms_ = np.sum(baseline_ms_, axis=1)

    ms_model_ = np.mean((ms_model_ - label_) ** 2, axis=2)
    ms_model_ = np.sum(ms_model_, axis=1)

    LATEX_SCALE =1
    lw = set_plot_style(FIG_WIDTH_IN*LATEX_SCALE)

    x = np.arange(1, steps + 1, 1)
    fig, axs = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.4))
    axs.plot(x, ms_model_, label="proposed", linewidth=lw)
    axs.plot(x, baseline_ms_, "--", label="baseline", linewidth=lw)
    axs.set_xlabel("prediction step")
    axs.set_ylabel("MSE(log)")
    axs.set_yscale("log")
    axs.set_ylim(1e-3, 2e-1)
    axs.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.3, 1.3))
    plt.subplots_adjust(left=0.2, right=0.99, top=0.85, bottom=0.3)
    plt.savefig("Cry_MSE.pdf", format="pdf", bbox_inches="tight")
    plt.show()


