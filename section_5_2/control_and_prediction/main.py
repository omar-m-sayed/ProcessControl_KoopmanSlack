import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt


# # Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.general.paths import PROJECT_ROOT
from utils.general.plot_helper import set_plot_style
from utils.utils_control.proposed_control_utils import proposedControl
from utils.utils_control.baseline_control_utils import BaselineControl
from utils.baseline_linearization_utils import LocalLinearization
from utils.utils_do_mpc_wrapper.do_mpc_polymerization import do_mpc_rapper_poly
from section_5_2.model_data.model_polymerization import PolymerizationSimulate

FIG_WIDTH_IN = 3.448

if __name__ == "__main__":
    """
    Load data, models and an instance of the crystallization model
    """
    data_path = os.path.join(PROJECT_ROOT, 'section_5_2', 'model_data', 'polymerization_dataset.pickle')

    with open(data_path, "rb") as handle:
        data = pickle.load(handle)

    ## Load Koopman models, initialize plant simulator and local linearization class
    data_path = os.path.join(PROJECT_ROOT, 'section_5_2', 'control_and_prediction', 'learned_models')
    proposed_model = torch.load(os.path.join(data_path, "proposed_polymerization_model"),weights_only=False)
    baseline_model = torch.load(os.path.join(data_path, "baseline_polymerization"), weights_only=False)
    polymerization_sim = PolymerizationSimulate(data["state_min"], data["state_max"],
                                                data["action_min"], data["action_max"])
    Local_linearization = LocalLinearization(polymerization_sim, data)

    """
    Prediction Comparison
    
    """

    input_step = np.array([[0.07], [0.93], [0.51], [0.05], [0.2]])
    inputs = np.repeat(input_step, 20, axis=0)
    states = data["state"][0, :, 0]
    label = polymerization_sim.get_trajectory(states.reshape(-1), inputs)
    linearized_ms = Local_linearization.get_ms_linearized(label[:-1, :], inputs)
    linearized_ms = np.hstack(
        [linearized_ms, (linearized_ms[:, 3] / linearized_ms[:, 2]).reshape(-1, 1)])  # Adding NAMW as last column

    states = torch.tensor(label[:-1, :]).float()
    inputs = torch.tensor(inputs).float()

    baseline_ms = baseline_model.get_ms_prediction(states, inputs).detach().numpy()
    proposed_model.initialize_prediction_optimization()  # must be initialized
    proposed_ms = proposed_model.get_ms_prediction(states, inputs).detach().numpy()

    nx = 5
    steps = 20

    label = label[1:, :]
    baseline_ms = baseline_ms
    proposed_ms = proposed_ms

    ## calculating MSE and MAE errors on normalized states

    mse_error_proposed = np.mean((label - proposed_ms) ** 2)
    mse_error_baseline = np.mean((label - baseline_ms) ** 2)
    mse_error_linearized = np.mean((label - polymerization_sim.get_regularized_state(linearized_ms)) ** 2, )
    mae_error_proposed = np.mean(np.sqrt((label - proposed_ms) ** 2))
    mae_error_baseline = np.mean(np.sqrt((label - baseline_ms) ** 2))
    mae_error_linearized = np.mean(np.sqrt((label - polymerization_sim.get_regularized_state(linearized_ms)) ** 2))

    label = polymerization_sim.get_real_state(label)
    baseline_ms = polymerization_sim.get_real_state(baseline_ms)
    proposed_ms = polymerization_sim.get_real_state(proposed_ms)

    time_grid = np.arange(0, label.shape[0]).reshape(-1, steps).T
    base = baseline_ms.reshape(-1, 1, nx).reshape(-1, 5, nx, order="F")
    proposed = proposed_ms.reshape(-1, 1, nx).reshape(-1, 5, nx, order="F")
    linearized = linearized_ms.reshape(-1, 1, nx).reshape(-1, 5, nx, order="F")

    label = label[:, [0, 2, 3, 4]]
    proposed = proposed[:, :, [0, 2, 3, 4]]
    base = base[:, :, [0, 2, 3, 4]]
    linearized = linearized[:, :, [0, 2, 3, 4]]
    time_grid = time_grid[:, :]

    lw = set_plot_style(FIG_WIDTH_IN)

    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 2.5))
    y_label = [r'$C_\mathrm{m}$ [kmol$\cdot$m$^{-3}$]', r'$\mathrm{D_0}$', r'$\mathrm{D_I}$',
               r'$\mathrm{NAMW}\left(\mathrm{D_I/D_0}\right)$']

    var_labels = y_label

    for i in range(4):
        row, col = divmod(i, 2)
        if i == 0:
            axs[row, col].plot(label[:, 1], label="true", color="green", linewidth=lw)
            axs[row, col].plot(time_grid[:, 0], proposed[:, 0, i], "-.", color="C0", linewidth=lw)
            axs[row, col].plot(time_grid[:, 0], base[:, 0, i], "--", color="C1", linewidth=lw)
            axs[row, col].plot(time_grid, linearized[:, :, i], "--", color="black", linewidth=lw)

            fig.legend(["true", "proposed", "baseline", "linearized"], ncol=3, bbox_to_anchor=(0.98, 1.03),
                       frameon=False)

        axs[row, col].plot(label[:, i], label="true", color="green", linewidth=lw)
        axs[row, col].plot(time_grid, proposed[:, :, i], "-.", color="C0", linewidth=lw)
        axs[row, col].plot(time_grid, base[:, :, i], "--", color="C1", linewidth=lw)
        axs[row, col].plot(time_grid, linearized[:, :, i], "--", color="black", linewidth=lw)

        if i == 0:
            axs[0, 0].set_ylim(5, 5.8)
        axs[row, col].set_ylabel(var_labels[i])
        if i <= 1:
            axs[row, col].set_xticks([])
        if i in [1, 2, 3]:
            axs[row, col].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[1, 0].set_xlabel("time steps")
    axs[1, 1].set_xlabel("time steps")
    for ax in [axs[0, 1], axs[1, 1]]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    for ax_row in axs:
        for ax in ax_row:
            for i in range(0, label.shape[0], steps):
                ax.axvline(x=i, color="black", linestyle='--', alpha=0.3, linewidth=lw)
    fig.align_ylabels(axs)

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0.1, hspace=0.2)
    plt.savefig("polymerization_prediction.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    """
    Control Comparison
    """

    ## setting up the koopman controllers

    nz = 10  ## number koopman states
    nu = 1  ## number of inputs
    steps = 20  # control horizion

    ## Extracting A, B, C matrices for proposed model
    A_proposed = proposed_model.msModels[:, :nz]
    A_proposed = torch.hstack(torch.split(A_proposed, nz, dim=0))  # Reshape flat A
    B_proposed = proposed_model.msModels[:, nz:]
    C_tilde_proposed = proposed_model.c

    ## Extracting A, B, C matrices for baseline model
    A_baseline = baseline_model.msModels[:, :nz]
    A_baseline = torch.hstack(torch.split(A_baseline, nz, dim=0))  # Reshape flat A
    B_baseline = baseline_model.msModels[:, nz:]
    C_baseline = baseline_model.c

    # same settings for both controllers
    x_min = np.array([0, 0.0, 0, 0, 0])
    x_max = np.array([1, 1, 1, 1, 1])
    u_min = np.array([0])
    u_max = np.array([1])
    Q = np.eye(5) * 0
    Q[4, 4] = 1  ## Tracking Namw
    R = np.eye(1) * 0.0

    control_proposed = proposedControl(A_proposed, B_proposed, C_tilde_proposed, proposed_model.NN_encoder, steps, nz,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=nu, real_model=polymerization_sim,
                                       model_decoder=proposed_model.NN_decoder, nx=nx)

    control_baseline = BaselineControl(A_baseline, B_baseline, C_baseline, baseline_model.NN_encoder, steps, nz,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=nu, real_model=polymerization_sim, nx=nx)
    #

    x_0 = torch.tensor(data["state"][100, :, 1]).float()  ## initial state
    reference_num = 4
    reference_points = np.array([0.72, 0.55, 0.80, 0.63])
    ref_idx = 4  ## controlled state
    sim_steps = 30
    _, plant_proposed, inputs_proposed, ref_traj, control_proposed_time = control_proposed.closed_loop_control(
        x_0, reference_points, reference_num, ref_idx, sim_steps)

    _, plant_baseline, inputs_baseline, _ = control_baseline.closed_loop_control(
        x_0, reference_points, reference_num, ref_idx, sim_steps)

    ss = np.zeros([reference_num, 5])
    ss[:, 4] = reference_points
    reference_points_real = polymerization_sim.get_real_state(ss)[:, 4]

    Nmpc = do_mpc_rapper_poly(
        reference_points_real, reference_num, sim_steps, data["state_min"],
        data["state_max"], data["action_min"], data["action_max"],
        horizion=20, Q=np.array([1]), R=np.array([0.0001])
    )
    n_mpc_x_0 = polymerization_sim.get_real_state(x_0)[:4].detach().numpy()
    states_nmpc, controls_nmpc, nmpc_time = Nmpc.get_mpc_trajectory(n_mpc_x_0)
    NAMW_mpc = states_nmpc[:, 3] / states_nmpc[:, 2]

    ## only for the local linearization
    linearized_initial_state = polymerization_sim.get_real_state(x_0)[:4].detach().numpy()
    ## since NAMW is not a state, we track d0 and d1
    Q = np.eye(4)
    Q[1, 1] = 0
    Q[0, 0] = 0
    R = np.eye(1) * 0

    ## utilitze mpc trajectory to get reference trajectories it comes from the real plant
    ref_lin = np.arange(sim_steps, reference_num * sim_steps, sim_steps - 1)
    reference_arr_linearized = states_nmpc[ref_lin, :]
    linearied_mpc_states, linearized_mpc_input, linearized_time_taken = Local_linearization.run_linear_mpc_tracking(x_0,
                                                                                                                    reference_arr_linearized,
                                                                                                                    Q,
                                                                                                                    R,
                                                                                                                    horizon=20,
                                                                                                                    sim_steps=sim_steps)

    NAMW_linearized = linearied_mpc_states[:, 3] / linearied_mpc_states[:, 2]

    ## Map the states back to there orignal scale
    plant_proposed = polymerization_sim.get_real_state(plant_proposed)[1:, :]
    plant_baseline = polymerization_sim.get_real_state(plant_baseline)[1:, :]
    inputs_proposed = polymerization_sim.get_real_action(inputs_proposed)
    inputs_baseline = polymerization_sim.get_real_action(inputs_baseline)
    ref_traj = polymerization_sim.get_real_state(ref_traj)
    state_min = data["state_min"]

    LATEX_SCALE =1
    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)
    x = np.arange(inputs_proposed.shape[0])
    fig, axs = plt.subplots(4, 1, figsize=(FIG_WIDTH_IN,4.8))
    alpha = 1
    # D0
    axs[0].plot(plant_baseline[:, 2], "--", linewidth=lw, label="baseline", color="C1", alpha=alpha)
    axs[0].plot(state_min[2] * np.ones_like(x), "--", color="red", linewidth=lw, label="constraint")
    axs[0].plot(states_nmpc[:, 2], ":", color="brown", linewidth=lw, label="NMPC")
    axs[0].plot(linearied_mpc_states[:, 2], "--", color="black", linewidth=lw, label="linearized", alpha=alpha)
    axs[0].plot(plant_proposed[:, 2], "-.", color="C0", linewidth=lw, label="proposed")

    axs[0].set_ylabel(r'$\mathrm{D_0}$')
    axs[0].set_xticks([])
    axs[0].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    # D1
    axs[1].plot(plant_baseline[:, 3], "--", linewidth=lw, color="C1", alpha=alpha)
    axs[1].plot(state_min[3] * np.ones_like(x), "--", color="red", linewidth=lw)
    axs[1].plot(states_nmpc[:, 3], ":", color="brown", linewidth=lw)
    axs[1].plot(linearied_mpc_states[:, 3], "--", color="black", linewidth=lw, alpha=alpha)
    axs[1].plot(plant_proposed[:, 3], "-.", linewidth=lw, color="C0")

    axs[1].set_ylabel(r'$\mathrm{D_I}$')
    axs[1].set_xticks([])

    # NAMW
    axs[2].step(x, ref_traj[:, 4], linewidth=lw, color="green", label="setpoint", alpha=0.6)
    axs[2].plot(plant_baseline[:, 4], "--", linewidth=lw, color="C1", alpha=alpha)
    axs[2].plot(NAMW_mpc, ":", color="brown", linewidth=lw)
    axs[2].plot(NAMW_linearized, "--", color="black", linewidth=lw, alpha=alpha)
    axs[2].plot(plant_proposed[:, 4], "-.", linewidth=lw, color="C0")

    axs[2].set_ylabel(r'$\mathrm{NAMW}~\left(\mathrm{D_I/D_0}\right)$')
    axs[2].set_xticks([])
    axs[2].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    # Control inputs
    alpha = 0.85
    axs[3].hlines([0.003, 0.06], 0, len(x), colors='r', linestyles='dashed', linewidth=lw)

    axs[3].step(x, inputs_baseline[:, 0], "--", color="C1", linewidth=lw, alpha=alpha)
    axs[3].step(x, linearized_mpc_input, color="black", linewidth=lw, alpha=0.6)
    axs[3].step(x, controls_nmpc[:, 0], ":", color="brown", linewidth=lw)
    axs[3].step(x, inputs_proposed[:, 0], "-.", linewidth=lw, color="C0")
    axs[3].set_ylabel(r'$F_{\mathrm{I}}~[\mathrm{m^{-3}/h}]$')
    axs[3].set_xlabel("time steps")
    axs[3].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    fig.align_ylabels(axs)

    fig.legend(ncol=3, bbox_to_anchor=(1.0, 1.02), frameon=False)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.91, bottom=0.07, wspace=0.22, hspace=0.18)
    plt.savefig("polymerization_control.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    print("\n=== Controller Performance Summary ===")
    print(f"Total time for proposed controller: {control_proposed_time:.4f} s")
    print(f"Total time for NMPC controller    : {nmpc_time:.4f} s")
    print(f"Total time for Linearized controller    : {linearized_time_taken:.4f} s")
    print(f"Number of time steps              : {int(sim_steps * reference_num)}")

    print("\n--- Average Computation Time per Step ---")
    print(f"Proposed controller               : {control_proposed_time / (sim_steps * reference_num):.4f} s/step")
    print(f"NMPC controller                   : {nmpc_time / (sim_steps * reference_num):.4f} s/step")
    print(f"Linearized controller                   : {linearized_time_taken / (sim_steps * reference_num):.4f} s/step")

    print("\n--- Prediction Error Metrics ---")
    print(f"MSE (Proposed (MSE))         : {mse_error_proposed:.6f}")
    print(f"MSE (Baseline (MSE))    : {mse_error_baseline:.6f}")
    print(f"MSE (Linearized (MSE))    : {mse_error_linearized:.6f}")

    print(f"MAE (Proposed controller (MAE))         : {mae_error_proposed:.6f}")
    print(f"MAE (Baseline (MAE) )    : {mae_error_baseline:.6f}")
    print(f"MAE (Linearized (MAE))    : {mae_error_linearized:.6f}")
    print("========================================\n")
