"""
Visualization for FACM model (1D).
"""

import numpy as np
import matplotlib.pyplot as plt

from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


def visualize_facm(
    z_samples,
    x_data,
    trajectories,
    cm_onestep,
    coupling_indices,
    save_path=None,
):
    """
    Visualize FACM in 1D.

    Plots:
    - Training coupling (gray): z -> x straight lines
    - Euler/Heun trajectories (blue): ODE integration from z to x
    - CM one-step (orange): z -> x_end (consistency sampler with 1 step)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS["figsize"])

    n_infer = trajectories.shape[1]
    n_steps = trajectories.shape[0] - 1
    t_space = np.linspace(0, 1, n_steps + 1)

    color_coupling = COLORS["coupling_train"]
    color_ode = "#4472C4"
    color_cm = "#ff7f0e"

    # Training coupling
    for i, j in enumerate(coupling_indices):
        ax.plot(
            [0, 1],
            [z_samples[i], x_data[j]],
            color=color_coupling,
            alpha=PLOT_PARAMS["line_alpha"],
            linewidth=PLOT_PARAMS["line_width"],
            zorder=1,
            label="Training coupling" if i == 0 else None,
        )

    # ODE trajectories
    for i in range(n_infer):
        ax.plot(
            t_space,
            trajectories[:, i],
            color=color_ode,
            alpha=0.7,
            linewidth=PLOT_PARAMS["line_width"] * 1.5,
            zorder=2,
            label="Euler/Heun ODE" if i == 0 else None,
        )

    # CM one-step
    z_infer = trajectories[0, :]
    for i in range(n_infer):
        ax.plot(
            [0, 1],
            [z_infer[i], cm_onestep[i]],
            color=color_cm,
            alpha=0.7,
            linewidth=PLOT_PARAMS["line_width"] * 2.0,
            zorder=2,
            label="CM one-step" if i == 0 else None,
        )

    # Points
    x0 = trajectories[0, :]
    x1_ode = trajectories[-1, :]
    x1_cm = cm_onestep

    ax.scatter(
        np.zeros_like(z_samples),
        z_samples,
        color=COLORS["source_x0"],
        marker="o",
        s=PLOT_PARAMS["point_size"],
        alpha=PLOT_PARAMS["point_alpha"],
        label="x₀ (train)",
        zorder=3,
    )
    ax.scatter(
        np.zeros(n_infer),
        x0,
        color=COLORS["source_x0_infer"],
        marker="o",
        s=PLOT_PARAMS["point_size"],
        alpha=PLOT_PARAMS["point_alpha"],
        label="x₀ (infer)",
        zorder=3,
    )
    ax.scatter(
        np.ones(n_infer),
        x1_ode,
        color=color_ode,
        marker="o",
        s=PLOT_PARAMS["point_size"],
        alpha=PLOT_PARAMS["point_alpha"],
        label="x₁ from ODE",
        zorder=3,
    )
    ax.scatter(
        np.ones(n_infer),
        x1_cm,
        color=color_cm,
        marker="s",
        s=PLOT_PARAMS["point_size"] * 1.2,
        alpha=PLOT_PARAMS["point_alpha"],
        label="x₁ from CM (1-step)",
        zorder=3,
    )
    ax.scatter(
        np.ones_like(x_data),
        x_data,
        color="black",
        marker=PLOT_PARAMS["marker_data_x"],
        s=PLOT_PARAMS["point_size"],
        alpha=PLOT_PARAMS["point_alpha"],
        label="x₁ (train data)",
        zorder=3,
    )

    all_values = np.concatenate([z_samples, x_data, trajectories.flatten(), cm_onestep])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS["y_padding"], y_max + PLOT_PARAMS["y_padding"])

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Value")
    ax.set_title("FACM Trajectory")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["t=0 (x₀)", "t=1 (x₁)"])
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"FACM visualization saved to {save_path}")

    return fig, ax

