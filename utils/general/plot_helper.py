import matplotlib as plt


def set_plot_style(fig_width_in, reference_width=3.448, base_fontsize=10, base_lw=1.6):

    scale = reference_width / fig_width_in

    plt.rcParams.update({
        "font.size": base_fontsize * scale,
        "axes.labelsize": base_fontsize * scale,
        "axes.titlesize": base_fontsize  * scale,
        "legend.fontsize": base_fontsize*0.84  * scale,
        "xtick.labelsize": base_fontsize * scale,
        "ytick.labelsize": base_fontsize  * scale,
        "lines.linewidth": base_lw * scale,
        'text.usetex': True,

    })

    return base_lw * scale