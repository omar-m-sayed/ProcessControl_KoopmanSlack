import os
import sys
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils.general.paths import PROJECT_ROOT
from utils.general.plot_helper import set_plot_style
from section_5_1.model_data.model import QuadrupleSim


def reshape_model_output(predictions, num_inputs, nx):
    return predictions.reshape(-1, 1, nx).reshape(-1, num_inputs, nx, order="F")


seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    data_path = os.path.join(PROJECT_ROOT, 'section_5_1', 'model_data', 'quadruple_tank.pickle')
    with open(data_path, "rb") as handle:
        training_data = pickle.load(handle)


    FIG_WIDTH_IN = 3.448
    # fixed parameters
    num_koopman_states = 6
    num_real_states = 4
    steps = 30

    # loading the models
    data_path = os.path.join(PROJECT_ROOT, 'section_5_1', 'results', 'prediction', 'learned_models')
    ss_model = torch.load(os.path.join(data_path, 'baseline_ss_30_steps'), weights_only=False)
    ms_model = torch.load(os.path.join(data_path, "baseline_ms_30_steps"), weights_only=False)
    ss_model_1 = torch.load(os.path.join(data_path, "baseline_ss_1_steps"), weights_only=False)

    # initializing the quad simulator
    Quad_sim = QuadrupleSim(training_data["state_min"], training_data["state_max"], training_data["action_min"],
                            training_data["action_max"])

    ## initial_state
    state = training_data["state"][1000, :, 0]
    s = np.random.randint(0, 100, [12, 2]) / 100
    action = np.repeat(s, 15, axis=0)

    # get the true simulation of the model
    real_traj = Quad_sim.get_trajectory(state, action)[:, :]
    state = real_traj[:-1, :]
    label = real_traj[1:, :]

    # compute the predictions
    ss_model_pred = ss_model.multi_step_prediction(torch.tensor(state).float(),
                                                   torch.tensor(action).float()).detach().numpy()
    ms_model_pred = ms_model.get_ms_prediction(torch.tensor(state).float(),
                                               torch.tensor(action).float()).detach().numpy()

    ## important to set the number of the open-loop prediction steps for the single step model
    ss_model_1.steps = steps
    ss_model_1_pred = ss_model_1.multi_step_prediction(torch.tensor(state).float(),
                                                       torch.tensor(action).float()).detach().numpy()

    ## number of prediction horizons
    n_predictions = np.ceil(action.shape[0] / steps)

    # Converting normalized values to the original values
    ss_model_plot = Quad_sim.get_real_state(ss_model_pred)
    ms_model_plot = Quad_sim.get_real_state(ms_model_pred)
    ss_model_1_plot = Quad_sim.get_real_state(ss_model_1_pred)
    label_plot = Quad_sim.get_real_state(label)

    # plotting the results
    ss_model_plot = reshape_model_output(ss_model_plot, num_koopman_states, num_real_states)
    ms_model_plot = reshape_model_output(ms_model_plot, num_koopman_states, num_real_states)
    ss_model_1_plot = reshape_model_output(ss_model_1_plot, num_koopman_states, num_real_states)

    x = np.arange(0, label.shape[0], 1).reshape(-1, steps)
    x = x.T
    LATEX_SCALE = 1

    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)

    fig, axs = plt.subplots(4, 1, figsize=(FIG_WIDTH_IN,4.8))
    plt.subplots_adjust(wspace=0.4, hspace=0.2)  # Adjust horizontal and vertical space

    axs[0].plot(label_plot[1:, 0], label="true", color="green", linewidth=lw)
    axs[0].plot(x[:, 0], ms_model_plot[:, 0, 0], "--", color="C0", linewidth=lw)
    axs[0].plot(x[:, 0], ss_model_plot[:, 0, 0], "-.", label="linear reconstruction", linewidth=lw, color="C1")
    axs[0].plot(x[:, 0], ss_model_1_plot[:, 0, 0], ":", label="linear reconstruction", linewidth=lw, color="C3")
    fig.legend(["true", "MP", "SS-30", "SS-1"], ncol=4, bbox_to_anchor=(1.02, 0.97), frameon=False)
    axs[0].plot(x[:, 1:], ms_model_plot[:, 1:, 0], "--", color="C0", linewidth=lw)
    axs[0].plot(x[:, 1:], ss_model_plot[:, 1:, 0], "-.", label="linear reconstruction", linewidth=lw, color="C1")
    axs[0].plot(x[:, 1:], ss_model_1_plot[:, 1:, 0], ":", label="linear reconstruction", linewidth=lw, color="C3")
    axs[0].set_xticks([])
    axs[0].set_ylabel(r'$h_1$[cm]')
    axs[1].plot(label_plot[1:, 1], label="True", color="green", linewidth=lw)
    axs[1].plot(x[:, :], ms_model_plot[:, :, 1], "--", color="C0", linewidth=lw)
    axs[1].plot(x[:, :], ss_model_plot[:, :, 1], "-.", label="linear reconstruction", linewidth=lw, color="C1")
    axs[1].plot(x[:, :], ss_model_1_plot[:, :, 1], ":", label="linear reconstruction", linewidth=lw, color="C3")
    axs[1].set_xticks([])
    axs[1].set_ylabel(r'$h_2$[cm]')
    axs[2].plot(label_plot[1:, 2], label="True", color="green", linewidth=lw)
    axs[2].plot(x, ms_model_plot[:, :, 2], "--", color="C0", linewidth=lw)
    axs[2].plot(x, ss_model_plot[:, :, 2], "-.", color="C1", linewidth=lw)
    axs[2].plot(x[:, :], ss_model_1_plot[:, :, 2], ":", label="linear reconstruction", linewidth=lw, color="C3")
    axs[2].set_ylabel(r'$h_3$[cm]')
    axs[2].set_xticks([])
    axs[3].plot(label_plot[1:, 3], label="True", color="green", linewidth=lw)
    axs[3].plot(x, ms_model_plot[:, :, 3], "--", color="C0", linewidth=lw)

    axs[3].plot(x, ss_model_plot[:, :, 3], "-.", color="C1", linewidth=lw)
    axs[3].plot(x[:, :], ss_model_1_plot[:, :, 3], ":", label="linear reconstruction", linewidth=lw, color="C3")

    axs[3].set_xlabel("time steps")
    axs[3].set_ylabel(r'$h_4$[cm]')

    ## plotting the vertical lines
    for ax in axs:
        for i in range(0, label_plot.shape[0] + steps, steps):
            if i == 0:
                ax.axvline(x=i, color='black', linestyle='--', alpha=0.3, linewidth=lw)
            else:
                ax.axvline(x=i - 1, color='black', linestyle='--', alpha=0.3, linewidth=lw)

    plt.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.12, wspace=0.32, hspace=0.1)
    fig.align_ylabels(axs)
    ##plt.savefig("Quadruple_prediction.pgf", format="pgf")
    plt.savefig("Quadruple_prediction.pdf", format="pdf",bbox_inches="tight")
    plt.show()
    ## print loss

    mse_mp = np.mean((ms_model_pred - label) ** 2)
    mse_ss_30 = np.mean((ss_model_pred - label) ** 2)
    mse_sss_model_1 = np.mean((ss_model_1_pred - label) ** 2)
    mae_mp = np.mean(np.abs(ms_model_pred - label))
    mae_ss_30 = np.mean(np.abs(ss_model_pred - label))
    mae_sss_model_1 = np.mean(np.abs(ss_model_1_pred - label))

    print("=== Model Evaluation Summary ===")
    print(f"MSE Loss (MS Model):              {mse_mp:.6f}")
    print(f"MSE Loss (SS Model, 30 steps):    {mse_ss_30:.6f}")
    print(f"MSE Loss (SS Model, 1 step):      {mse_sss_model_1:.6f}")
    print(f"MAE Loss (MS Model):              {mae_mp:.6f}")
    print(f"MAE Loss (SS Model, 30 steps):    {mae_ss_30:.6f}")
    print(f"MAE Loss (SS Model, 1 step):      {mae_sss_model_1:.6f}")

    state = training_data["state"][1000, :, 0]

    ## testing the models for 100 open loop predictions
    s = np.random.randint(0, 100, [200, 2]) / 100
    action = np.repeat(s, 15, axis=0)


    real_traj = Quad_sim.get_trajectory(state, action)[:, :]
    state = real_traj[:-1, :]
    label = real_traj[1:, :]
    ss_model = ss_model.multi_step_prediction(torch.tensor(state).float(),
                                              torch.tensor(action).float()).detach().numpy()
    ms_model = ms_model.get_ms_prediction(torch.tensor(state).float(), torch.tensor(action).float()).detach().numpy()
    ss_model_1.steps = steps
    ss_model_1 = ss_model_1.multi_step_prediction(torch.tensor(state).float(),
                                                  torch.tensor(action).float()).detach().numpy()

    label_ = np.reshape(label, (steps, -1, 4), order="F")
    label_ = label_.transpose(0, 2, 1)

    ss_model_ = ss_model.reshape(steps, -1, num_real_states, order="F")
    ss_model_ = ss_model_.transpose(0, 2, 1)
    ms_model_ = ms_model.reshape(steps, -1, num_real_states, order="F")
    ms_model_ = ms_model_.transpose(0, 2, 1)
    ss_model_1_ = ss_model_1.reshape(steps, -1, num_real_states, order="F")
    ss_model_1_ = ss_model_1_.transpose(0, 2, 1)

    ss_model_ = np.mean((ss_model_ - label_) ** 2, axis=2)
    ss_model_ = np.sum(ss_model_, axis=1)

    ss_model_1_ = np.mean((ss_model_1_ - label_) ** 2, axis=2)
    ss_model_1_ = np.sum(ss_model_1_, axis=1)
    ms_model_ = np.mean((ms_model_ - label_) ** 2, axis=2)
    ms_model_ = np.sum(ms_model_, axis=1)

    LATEX_SCALE = 1
    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)

    x = np.arange(1, steps + 1, 1)
    fig, axs = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, 1.4))
    axs.plot(x, ms_model_, label="MP", linewidth=lw)
    axs.plot(x, ss_model_, "--", label="SS-30", linewidth=lw)
    axs.plot(x, ss_model_1_, "-.", label="SS-1", color="C3", linewidth=lw)
    axs.set_xlabel("prediction step")
    axs.set_ylabel("MSE(log)")
    axs.set_yscale("log")
    fig.legend(frameon=False, ncol=3, bbox_to_anchor=(0.84, 1.02))
    plt.subplots_adjust(left=0.2, right=0.99, top=0.85, bottom=0.2)
    plt.savefig("Quadruple_MSE.pdf", format="pdf", bbox_inches="tight")
    plt.show()
