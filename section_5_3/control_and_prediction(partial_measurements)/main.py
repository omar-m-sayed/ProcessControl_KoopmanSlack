import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Path setup
# --------------------------------------------------------------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils'))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.general.paths import PROJECT_ROOT
from section_5_3.model_data.model_crystallization import CrystallizationSimulate
from utils.utils_control.proposed_control_utils import proposedControlNarx as proposedControl
from utils.utils_control.baseline_control_utils import BaselineControlNarx as BaselineControl
from utils.utils_do_mpc_wrapper.do_mpc_crystallization import do_mpc_rapper_crystalization
from utils.general.plot_helper import set_plot_style

if __name__ == "__main__":
    # --- Config---
    FIG_WIDTH_IN = 3.448  # in inches for single column
    IDX_T = 1  # column for temperature T
    IDX_L10 = 6  # column for L10

    # --- Load Data ---
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'model_data', 'Crystillization_noise.pickle')
    with open(data_path, "rb") as handle:
        sim_data = pickle.load(handle)
        data = {"state": sim_data["state"][:, [IDX_T, IDX_L10]], "label": sim_data["label"][:, [IDX_T, IDX_L10]],
                "action": sim_data["action"]}

    # --- Create simulation environment ---
    crystallization_sim = CrystallizationSimulate(sim_data["state_min"], sim_data["state_max"],
                                                  sim_data["action_min"], sim_data["action_max"])

    # --- plot noisy measurements vs True---
    UPT0 = 750
    START = 600
    LATEX_SCALE = 1

    orig_data = crystallization_sim.get_real_state(sim_data["state"][START:UPT0, :, 0])
    noisy_data = crystallization_sim.get_real_state(sim_data["state_noise"][START:UPT0, :, 0])
    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)

    fig, axs = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, 2.4), linewidth=lw)
    axs[0].plot(orig_data[:, 1], label='true', linewidth=lw)
    axs[0].plot(noisy_data[:, 1], label='measurements', linewidth=lw, alpha=0.7)
    axs[0].set_ylabel('$T$[K]')
    axs[0].set_xticks([])
    fig.legend(frameon=False, ncol=3, bbox_to_anchor=(0.7, 1.02))

    axs[1].plot(orig_data[:, 6], label='True State 2', linewidth=lw)
    axs[1].plot(noisy_data[:, 6], label='Noise State 2', linewidth=lw, alpha=0.7)
    axs[1].set_ylabel('$L_{10}$[m]')
    axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axs[1].set_xlabel('time steps')

    plt.subplots_adjust(left=0.18, right=0.95, top=0.91, bottom=0.2, wspace=0.2, hspace=0.25)
    fig.align_ylabels(axs)
    plt.savefig("noisy_measurements.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # --- Load trained models ---

    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'control_and_prediction(partial_measurements)', 'learned_models')
    proposed_model = torch.load(os.path.join(data_path, "Narx_proposed"), weights_only=False)
    baseline_model = torch.load(os.path.join(data_path, "Narx_baseline"), weights_only=False)
    proposed_model.NN_decoder.eval()

    # --- Multi-step prediction comparison ---
    # parameters
    proposed_dy = proposed_model.states_delay
    proposed_du = proposed_model.control_delay
    baseline_dy = baseline_model.states_delay
    baseline_du = baseline_model.control_delay
    NU = 2  # number of inputs
    HORIZON = 20
    NX = 2  # number of states

    idx = np.max([proposed_dy, baseline_dy, proposed_du, baseline_du])  # used for initial control

    test_input = np.array([[0.28360323, 0.6565945],
                           [0.85812413, 0.21386546],
                           [0.73955406, 0.66567724],
                           [0.96417531, 0.03021013],
                           [0.68949327, 0.48671752], ])

    initial_input = np.repeat(test_input[0, :], idx, axis=0).reshape(-1, NU)

    test_input = np.repeat(test_input[1:, ], HORIZON, axis=0)
    inputs = np.vstack((initial_input, test_input))
    states = sim_data["state"][200, :, 0]

    real_traj = crystallization_sim.get_trajectory(states.reshape(-1), inputs)
    labels = torch.tensor(real_traj[1:, :]).float()[:, [IDX_T, IDX_L10]]

    states = torch.tensor(real_traj[:-1, :]).float()[:, [IDX_T, IDX_L10]]
    inputs = torch.tensor(inputs[:, :]).float()

    # --- slice according to delays ---
    prop_offset = idx - max(proposed_dy, proposed_du)
    base_offset = idx - max(baseline_dy, baseline_du)

    states_proposed = states[prop_offset:, :]
    actions_proposed = inputs[prop_offset:, :]
    labels_proposed = labels[prop_offset:, :]

    states_baseline = states[base_offset:, :]
    actions_baseline = inputs[base_offset:, :]
    labels_baseline = labels[base_offset:, :]

    inputs = torch.tensor(inputs).float()
    proposed_model.initialize_prediction_optimization()

    # get multi-step predictions
    proposed_ms = proposed_model.get_ms_prediction(states_proposed, actions_proposed,
                                                   labels_proposed).detach().numpy()
    baseline_ms = baseline_model.get_ms_prediction(states_baseline, actions_baseline).detach().numpy()

    real_traj = crystallization_sim.get_real_state(real_traj[(idx + 1):, :])
    proposed_ms = crystallization_sim.get_real_state(proposed_ms, [IDX_T, IDX_L10])
    baseline_ms = crystallization_sim.get_real_state(baseline_ms, [IDX_T, IDX_L10])

    # Disconnect predictions for each prediction horizon
    time_grid = np.arange(0, real_traj.shape[0]).reshape(-1, HORIZON).T
    base = baseline_ms.reshape(-1, 1, NX).reshape(-1, 4, NX, order="F")
    proposed = proposed_ms.reshape(-1, 1, NX).reshape(-1, 4, NX, order="F")

    lw = set_plot_style(FIG_WIDTH_IN)
    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, 1.45))

    axs[0].plot(real_traj[:, 1], label="true", color="green", linewidth=lw)
    axs[0].plot(time_grid[:, 0], proposed[:, 0, 0], "-.", label="proposed", color="C0", linewidth=lw)
    axs[0].plot(time_grid[:, 0], base[:, 0, 0], "--", label="baseline", color="C1", linewidth=lw)
    fig.legend(["true", "proposed", "baseline"], ncol=3, bbox_to_anchor=(0.88, 1.05), frameon=False)

    axs[0].plot(real_traj[:, 1], label="True", color="green", linewidth=lw)
    axs[0].plot(time_grid, proposed[:, :, 0], "-.", color="C0", linewidth=lw)
    axs[0].plot(time_grid, base[:, :, 0], "--", color="C1", linewidth=lw)
    axs[0].set_ylabel('$T$[K]')
    axs[0].set_xlabel('time steps')

    axs[1].plot(real_traj[:, 6], label="True", color="green", linewidth=lw)
    axs[1].plot(time_grid, proposed[:, :, 1], "-.", color="C0", linewidth=lw)
    axs[1].plot(time_grid, base[:, :, 1], "--", color="C1", linewidth=lw)
    axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axs[1].set_ylabel('$L_{10}$[m]')
    axs[1].set_xlabel('time steps')
    fig.align_ylabels(axs)

    for i in range(0, real_traj.shape[0], 20):
        axs[0].axvline(x=i, color='black', linestyle='--', alpha=0.3, linewidth=lw)
        axs[1].axvline(x=i, color='black', linestyle='--', alpha=0.3, linewidth=lw)

    axs[1].yaxis.set_label_position("right")
    axs[1].yaxis.tick_right()
    plt.subplots_adjust(left=0.15, right=0.85, top=0.78, bottom=0.28, wspace=0.1, hspace=0.2)
    plt.savefig("narx_prediction.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # --- Closed-loop control comparison ---
    #  --- Config---
    KOOPMAN_SPACE_PROPOSED = KOOPMAN_SPACE_BASELINE = 10
    REF_IDX = 1  # index of reference in the state vector
    SIM_STEPS = 40
    NUM_REF = 3  # number of reference changes
    reference_points = np.array([0.76, 0.55, 0.82])

    ## extracting linear models
    A_proposed = proposed_model.msModels[:, :KOOPMAN_SPACE_PROPOSED]
    A_proposed = torch.hstack(torch.split(A_proposed, KOOPMAN_SPACE_PROPOSED, dim=0))  # Reshape flat A
    B_proposed = proposed_model.msModels[:, KOOPMAN_SPACE_PROPOSED:]
    C_tilde_proposed = proposed_model.c

    A_baseline = baseline_model.msModels[:, :KOOPMAN_SPACE_BASELINE]
    A_baseline = torch.hstack(torch.split(A_baseline, KOOPMAN_SPACE_BASELINE, dim=0))  # Reshape flat A
    B_baseline = baseline_model.msModels[:, KOOPMAN_SPACE_BASELINE:]
    C_baseline = baseline_model.c

    # state and input constraints (proposed & baseline)
    x_min = np.array([0.0, 0.0])
    x_max = np.array([1, 1])
    u_min = np.array([0, 0])
    u_max = np.array([1, 1])

    # Q and R (proposed & baseline)
    Q = np.eye(2) * 0
    Q[1, 1] = 1  # Tracking L_10
    R = np.eye(2) * 0

    control_proposed = proposedControl(A_proposed, B_proposed, C_tilde_proposed, proposed_model.NN_encoder, HORIZON,
                                       KOOPMAN_SPACE_PROPOSED,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=NU, real_model=crystallization_sim,
                                       model_decoder=proposed_model.NN_decoder, nx=NX, states_delay=proposed_dy,
                                       control_delay=proposed_du)

    control_baseline = BaselineControl(A_baseline, B_baseline, C_baseline, baseline_model.NN_encoder, HORIZON,
                                       KOOPMAN_SPACE_BASELINE,
                                       x_max=x_max, x_min=x_min,
                                       u_max=u_max, u_min=u_min, Q=Q, R=R, nu=NU, real_model=crystallization_sim,
                                       nx=NX,
                                       states_delay=baseline_dy, control_delay=baseline_du,
                                       eps_abs=1e-2, eps_rel=1e-2)

    initial_point_idx = 1500

    proposed_state_initial_point = data["state"][initial_point_idx - proposed_dy - 1:initial_point_idx, :, 1]
    baseline_state_initial_point = data["state"][initial_point_idx - baseline_dy - 1:initial_point_idx, :, 1]

    proposed_input_initial_point = data["action"][initial_point_idx - proposed_du:initial_point_idx, :, 1]
    baseline_input_initial_point = data["action"][initial_point_idx - baseline_du:initial_point_idx, :, 1]

    x_0 = sim_data["state"][initial_point_idx, :, 1]  # used for simulation

    proposed_plant, proposed_control, ref_traj, _ = control_proposed.closed_loop_control(
        x_0, proposed_state_initial_point, proposed_input_initial_point, reference_points, NUM_REF, REF_IDX,
        SIM_STEPS)

    baseline_plant, baseline_control, _ = control_baseline.closed_loop_control(
        x_0, baseline_state_initial_point, baseline_input_initial_point, reference_points, NUM_REF, REF_IDX,
        SIM_STEPS)
   
   
    # --- Convert to real states and actions (scaling) --- 
    proposed_plant_real = crystallization_sim.get_real_state(proposed_plant)
    baseline_plant_real = crystallization_sim.get_real_state(baseline_plant)
    baseline_control_real = crystallization_sim.get_real_action(baseline_control)
    proposed_control_real = crystallization_sim.get_real_action(proposed_control)

    temp = np.zeros([NUM_REF, 7])
    temp[:, 6] = reference_points
    reference_points_real = crystallization_sim.get_real_state(temp)[:, 6]

    # x_0 = sim_data["state"][initial_point_idx, :, 1]

    NMPC = do_mpc_rapper_crystalization(reference_points_real, NUM_REF, SIM_STEPS, sim_data["state_min"],
                                        sim_data["state_max"],
                                        sim_data["action_min"], sim_data["action_max"], horizion=HORIZON,
                                        Q=np.array([1000]),
                                        R=np.array([8e-4, 4e-4, ], dtype=float))


    x_0_real = crystallization_sim.get_real_state(x_0)[:6]
    states_NMPC, controls_NMPC, NMPC_time = NMPC.get_mpc_trajectory(x_0_real)
    L_10_NMPC = states_NMPC[:, 4] / states_NMPC[:, 3]

    contraint_min = sim_data["state_min"]
    ref_traj = crystallization_sim.get_real_state(ref_traj, [IDX_T, IDX_L10])

    LATEX_SCALE=1
    lw = set_plot_style(FIG_WIDTH_IN * LATEX_SCALE)
    x = np.arange(proposed_control.shape[0])


    fig, axs = plt.subplots(4, 1, figsize=(FIG_WIDTH_IN, 4.8))
    # --- Plotting of the control results ---

    axs[0].plot(baseline_plant_real [:, 1], "--", linewidth=lw, label="baseline", color="C1", alpha=0.75)
    axs[0].plot(contraint_min[1] * np.ones_like(x), "--", color="red", linewidth=lw, label="constraint")
    axs[0].plot(states_NMPC[:, 1], ":", color="brown", linewidth=lw, label="NMPC")
    axs[0].plot(proposed_plant_real[:, 1], "-.", color="C0", linewidth=lw, label="proposed")
    axs[0].set_ylabel(r'$T{\mathrm{[K]}}$')
    axs[0].set_xticks([])
    axs[0].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[1].plot( baseline_plant_real [:, 6], "--", linewidth=lw, color="C1", alpha=0.75)
    axs[1].plot(L_10_NMPC, ":", color="brown", linewidth=lw)
    axs[1].plot(proposed_plant_real[:, 6], "-.", linewidth=lw, color="C0")
    axs[1].step(x, ref_traj[:, 1], linewidth=lw, color="green", label="setpoint", alpha=0.6)

    axs[1].set_ylabel(r'$L_{\mathrm{10}}{\mathrm{[m]}}$')
    axs[1].set_xticks([])
    axs[1].ticklabel_format(style='scientific', axis='y', scilimits=(-3, 3))

    axs[2].hlines(0.1, 0, baseline_control_real.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[2].hlines(0.5, 0, baseline_control_real.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[2].plot(x, baseline_control_real[:, 0], "--", color="C1", linewidth=lw, alpha=0.5)
    axs[2].plot(x, proposed_control_real[:, 0], "-.", linewidth=lw, color="C0")
    axs[2].plot(x, controls_NMPC[:, 0], ":", color="brown", linewidth=lw)

    axs[2].set_ylabel(r'$F_{\mathrm{feed}}{\mathrm{[m/s]}}$')
    axs[2].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    axs[2].set_xticks([])
    #
    axs[3].hlines(0.1, 0, baseline_control_real.shape[0], colors='r', linestyles='dashed', linewidth=lw)
    axs[3].hlines(0.5, 0, baseline_control_real.shape[0], colors='r', linestyles='dashed', linewidth=lw)

    axs[3].plot(x, baseline_control_real[:, 1], "--", color="C1", linewidth=lw, alpha=0.5)
    axs[3].plot(x, proposed_control_real[:, 1], "-.", linewidth=lw, color="C0")
    axs[3].plot(x, controls_NMPC[:, 1], ":", color="brown", linewidth=lw)

    axs[3].set_ylabel(r'$F_{\mathrm{J}}{\mathrm{[m/s]}}$')
    axs[3].set_xlabel("time steps")
    axs[3].ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))

    fig.legend(frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.01))
    fig.align_ylabels(axs)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.91, bottom=0.1, wspace=0.2, hspace=0.2)
    plt.savefig("narx_control.pdf", format="pdf", bbox_inches="tight")
    plt.show()
