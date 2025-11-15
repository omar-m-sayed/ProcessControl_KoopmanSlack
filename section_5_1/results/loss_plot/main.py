
import os
import sys
# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import pickle

import matplotlib.pyplot as plt
from utils.general.paths import PROJECT_ROOT
from utils.general.plot_helper import set_plot_style



steps = [1, 10, 15, 20, 30,40,50]

colors = plt.get_cmap('tab10', len(steps))
FIG_WIDTH_IN = 3.448






lw = set_plot_style(FIG_WIDTH_IN )
fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 2.4))
training_setting = "Rprop_Tanh"
## plot training loss for statespace and multistep predictors for different training setting
data_path = os.path.join(PROJECT_ROOT, 'section_5_1', 'results', 'loss_plot')

for i, step in enumerate(steps):
    with open(os.path.join(data_path,f"{training_setting}", "ss", f"baseline_ss_{step}_steps.pkl"), "rb") as handle:
        loss_ss = pickle.load(handle)
    with open(os.path.join(data_path,f"{training_setting}", "ms", f"baseline_ms_{step}_steps.pkl"), "rb") as handle:
        loss_ms = pickle.load(handle)

    color = colors(i)

    if step==1:
        ax.plot(loss_ms, linestyle='--', color=color, linewidth=lw)
        # since both ss and ms will use the least squares solution, thus ms and ss are exactly the same
        ax.plot(loss_ms, label=f"{step}-step", color=color, linewidth=lw)
    else:
        ax.plot(loss_ss, linestyle='--', color=color, linewidth=lw)
        ax.plot(loss_ms, label=f"{step}-steps", color=color, linewidth=lw)


    # Print final loss values
    final_ss = loss_ss[-1]
    final_ms = loss_ms[-1]
    print(f"{step:>5} | {final_ss:>15.6f} | {final_ms:>15.6f}")

# Plot formatting
ax.set_yscale('log')
ax.set_xlabel("epochs")
ax.set_ylabel("MSE(log)")
#ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.45, 1.55), frameon=False)
fig.legend(ncol=3, bbox_to_anchor=(0.95, 1.02), frameon=False )
fig.subplots_adjust(top=0.78, left=0.18,bottom=0.15,right=0.99)
plt.savefig("Quadruple_loss.pdf", format="pdf", bbox_inches="tight")
plt.show()
