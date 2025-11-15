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
from utils.utils_control.proposed_control_utils import proposedControl
from utils.utils_control.baseline_control_utils import BaselineControl
from utils.utils_do_mpc_wrapper.do_mpc_crystallization import do_mpc_rapper_crystalization
from section_5_3.model_data.model_crystallization import CrystallizationSimulate

FIG_WIDTH_IN = 3.448

if __name__ == "__main__":
    """
    Load data, models and an instance of the crystallization model
    """
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'model_data', 'crystallization_data.pickle')
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    ## load the models
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'control_and_prediction', 'learned_models')
    proposed_model = torch.load(os.path.join(data_path, "proposed_crystallization_model"),weights_only=False)
    baseline_model = torch.load(os.path.join(data_path, "baseline_crystallization_model"),weights_only=False)
    crystallization_sim = CrystallizationSimulate(data["state_min"], data["state_max"],
                                                  data["action_min"], data["action_max"])

    ## Testing step Inputs
    input_step = np.array([[0.68, 0.1],
                           [0.31, 0.31],
                           [0.45, 0.66],
                           [0.99, 0.06],
                           [0.2, 0.77]])
    inputs = np.repeat(input_step, 20, axis=0)
    states = data["state"][0, :, 0]
    label = crystallization_sim.get_trajectory(states.reshape(-1), inputs)

    states = torch.tensor(label[:-1, :]).float()
    inputs = torch.tensor(inputs).float()

    baseline_ms = baseline_model.get_ms_prediction(states, inputs).detach().numpy()
    proposed_model.initialize_prediction_optimization()
    proposed_ms = proposed_model.get_ms_prediction(states, inputs).detach().numpy()

    nx = 7

    label = label[1:, :]
    baseline_ms = baseline_ms
    proposed_ms = proposed_ms

    ## calculating MSE and MAE errors on normalized states
    mse_error_proposed = np.mean((label - proposed_ms) ** 2, )
    mse_error_baseline = np.mean((label - baseline_ms) ** 2, )
    mae_error_proposed = np.mean(np.sqrt((label - proposed_ms) ** 2))
    mae_error_baseline = np.mean(np.sqrt((label - baseline_ms) ** 2))

    label = crystallization_sim.get_real_state(label)
    baseline_ms = crystallization_sim.get_real_state(baseline_ms)
    proposed_ms = crystallization_sim.get_real_state(proposed_ms)

    time_grid = np.arange(0, label.shape[0]).reshape(-1, 20).T
    base = baseline_ms.reshape(-1, 1, nx).reshape(-1, 5, nx, order="F")
    proposed = proposed_ms.reshape(-1, 1, nx).reshape(-1, 5, nx, order="F")


    label = label[:, [0, 1, 4, 6]]
    proposed = proposed[:, :, [0, 1, 4, 6]]
    base = base[:, :, [0, 1, 4, 6]]
    lw = set_plot_style(FIG_WIDTH_IN)

    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 2.5))
    var_labels = [r'$c$', r'$T$[K]', r'$\mu_1$[m]', r'$L_{10}$[m]']
    for i in range(4):
        row, col = divmod(i, 2)
        if i == 0:
            axs[row, col].plot(label[:, i], label="True", color="green", linewidth=lw)
            axs[row, col].plot(time_grid[:, 0], proposed[:, 0, i], "-.", color="C0", linewidth=lw)
            axs[row, col].plot(time_grid[:, 0], base[:, 0, i], "--", color="C1", linewidth=lw)
        axs[row, col].plot(label[:, i], label="True", color="green", linewidth=lw)
        axs[row, col].plot(time_grid, proposed[:, :, i], "-.", color="C0", linewidth=lw)
        axs[row, col].plot(time_grid, base[:, :, i], "--", color="C1", linewidth=lw)
        axs[row, col].set_ylabel(var_labels[i])
        if (row < 1):
            axs[row, col].set_xticks([])
        if i in [2, 4, 3]:
            axs[row, col].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[1, 0].set_xlabel("time steps")
    axs[1, 1].set_xlabel("time steps")
    for ax in [axs[0, 1], axs[1, 1]]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()


    # fig.delaxes(axs[2, 1])

    for ax_row in axs:
        for ax in ax_row:
            for i in range(0, label.shape[0], 20):
                ax.axvline(x=i, color='black', linestyle='--', alpha=0.3, linewidth=lw)

    fig.legend(["true", "proposed", "baseline"], ncol=3, bbox_to_anchor=(0.9, 0.96), frameon=False)
    fig.align_ylabels(axs)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0.1, hspace=0.2)
    plt.savefig("Crystalization_prediction.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    ## setting up the koopman controllers

    nz = 10  ## number koopman states
    nu = 2  ## number of inputs
    steps = 20  # control horizion

    ## Extracting A, B, C matrices for proposed model
    A_proposed = proposed_model.msModels[:, :nz]
    A_proposed = torch.hstack(torch.split(A_proposed, nz, dim=0))  # Reshape flat A
    B_proposed = proposed_model.msModels[:, nz:]
    C_tilde_proposed = proposed_model.c
    nz = 10  ## number koopman states

    ## Extracting A, B, C matrices for baseline model
    A_baseline = baseline_model.msModels[:, :nz]
    A_baseline = torch.hstack(torch.split(A_baseline, nz, dim=0))  # Reshape flat A
    B_baseline = baseline_model.msModels[:, nz:]
    C_baseline = baseline_model.c

    ## same configurations for both controllers

    x_min = np.array([0, 0.0, 0, 0, 0, 0, 0])
    x_max = np.array([1, 1, 1, 1, 1, 1, 1])
    u_min = np.array([0, 0])
    u_max = np.array([1, 1])
    Q = np.eye(7) * 0
    Q[6, 6] = 1  ## Tracking L10
    R = np.eye(2) * 0.0

    control_proposed = proposedControl(A_proposed, B_proposed, C_tilde_proposed, proposed_model.NN_encoder, steps, nz,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=nu, real_model=crystallization_sim,
                                       model_decoder=proposed_model.NN_decoder, nx=nx)

    control_baseline = BaselineControl(A_baseline, B_baseline, C_baseline, baseline_model.NN_encoder, 20, nz,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=2, real_model=crystallization_sim, nx=nx,
                                       eps_abs=1e-1,
                                       eps_rel=1e-0)  ## tolerance settings for the baseline controller canot be increased
    ## due to model mismatch

    x_0 = data["state"][1000, :, 0]
    ref_idx = 6
    sim_steps = 50
    reference_num = 3
    reference_points = np.array([0.85, 0.2, 0.75])

    _, plant_proposed, inputs_proposed, ref_traj, control_proposed_time = control_proposed.closed_loop_control(
        x_0, reference_points, reference_num, ref_idx, sim_steps)

    _, plant_baseline, inputs_baseline, _, = control_baseline.closed_loop_control(
        x_0, reference_points, reference_num, ref_idx, sim_steps)

    ss = np.zeros([reference_num, 7])
    ss[:, 6] = reference_points
    reference_points_real = crystallization_sim.get_real_state(ss)[:, 6]

    ## solving for the optimal controls using NMPC
    Nmpc = do_mpc_rapper_crystalization(reference_points_real, reference_num, sim_steps, data["state_min"],
                                        data["state_max"],
                                        data["action_min"], data["action_max"], horizion=20,
                                        Q=np.array([1000]),
                                        R=np.array([2e-4, 4e-4, ], dtype=float))

    linearized_initial_state = crystallization_sim.get_real_state(x_0)[:6]

    states_nmpc, controls_nmpc, nmpc_time = Nmpc.get_mpc_trajectory(linearized_initial_state)

    ## converting back to the orignal scale
    plant_proposed = crystallization_sim.get_real_state(plant_proposed)[1:, :]
    plant_baseline = crystallization_sim.get_real_state(plant_baseline)[1:, :]
    inputs_proposed = crystallization_sim.get_real_action(inputs_proposed)
    inputs_baseline = crystallization_sim.get_real_action(inputs_baseline)
    ref_traj = crystallization_sim.get_real_state(ref_traj)
    contraint_min = data["state_min"]

    L_10 = states_nmpc[:, 4] / states_nmpc[:, 3]


    LATEX_SCALE = 1
    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)

    fig, axs = plt.subplots(4, 1, figsize=(FIG_WIDTH_IN, 4.8))
    x = np.arange(inputs_proposed.shape[0])

    # D0
    axs[0].plot(plant_baseline[:, 1], "--", linewidth=lw, label="baseline", color="C1", alpha=0.75)
    axs[0].plot(contraint_min[1] * np.ones_like(x), "--", color="red", linewidth=lw, label="constraint")
    axs[0].plot(states_nmpc[:, 1], ":", color="brown", linewidth=lw  , label="NMPC")
    axs[0].plot(plant_proposed[:, 1], "-.", color="C0", linewidth=lw, label="proposed")

    axs[0].set_ylabel(r'$T{\mathrm{[K]}}$')
    axs[0].set_xticks([])
    axs[0].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[1].plot(plant_baseline[:, 6], "--", linewidth=lw, color="C1", alpha=0.75)
    axs[1].plot(L_10, ":", color="brown", linewidth=lw )
    axs[1].plot(plant_proposed[:, 6], "-.", linewidth=lw, color="C0")
    axs[1].step(x, ref_traj[:, 6], linewidth=lw, color="green", label="setpoint", alpha=0.6)

    axs[1].set_ylabel(r'$L_{\mathrm{10}}{\mathrm{[m]}}$')
    axs[1].set_xticks([])
    axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[2].hlines(0.1, 0, inputs_baseline.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[2].hlines(0.8, 0, inputs_baseline.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[2].plot(x, inputs_baseline[:, 0], "--", color="C1", linewidth=lw, alpha=0.5)
    axs[2].plot(x, inputs_proposed[:, 0], "-.", linewidth=lw, color="C0")
    axs[2].plot(x, controls_nmpc[:, 0], ":", color="brown", linewidth=lw)

    axs[2].set_ylabel(r'$F_{\mathrm{feed}}{\mathrm{[m/s]}}$')
    axs[2].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axs[2].set_xticks([])

    axs[3].hlines(0.1, 0, inputs_baseline.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[3].hlines(0.8, 0, inputs_baseline.shape[0], colors='r', linestyles='dashed', linewidth=lw)

    axs[3].plot(x, inputs_baseline[:, 1], "--", color="C1", linewidth=lw, alpha=0.5)
    axs[3].plot(x, inputs_proposed[:, 1], "-.", linewidth=lw, color="C0")
    axs[3].plot(x, controls_nmpc[:, 1], ":", color="brown", linewidth=lw)

    axs[3].set_ylabel(r'$F_{\mathrm{J}}{\mathrm{[m/s]}}$')
    axs[3].set_xlabel("time steps")
    axs[3].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))

    # Layout
    fig.legend(frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.01))
    fig.align_ylabels(axs)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.91, bottom=0.1, wspace=0.2, hspace=0.2)
    plt.savefig("Crystalization_control.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    print("\n=== Controller Performance Summary ===")
    print(f"Total time for proposed controller: {control_proposed_time:.4f} s")
    print(f"Total time for NMPC controller    : {nmpc_time:.4f} s")
    print(f"Number of time steps              : {int(sim_steps * reference_num)}")

    print("\n--- Average Computation Time per Step ---")
    print(f"Proposed controller               : {control_proposed_time / (sim_steps * reference_num):.4f} s/step")
    print(f"NMPC controller                   : {nmpc_time / (sim_steps * reference_num):.4f} s/step")

    print("\n--- Prediction Error Metrics ---")
    print(f"MSE (Proposed (MSE))         : {mse_error_proposed:.6f}")
    print(f"MSE (Baseline (MSE))    : {mse_error_baseline:.6f}")
    print(f"MAE (Proposed (MAE)))         : {mae_error_proposed:.6f}")
    print(f"MAE (Baseline (MAE))    : {mae_error_baseline:.6f}")
    print("========================================\n")
