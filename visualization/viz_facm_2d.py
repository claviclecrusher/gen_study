"""
Visualization for FACM model in 2D.
"""

import matplotlib.pyplot as plt

from visualization.config import COLORS, setup_plot_style


def visualize_facm_2d(
    trajectories,
    cm_onestep,
    z_samples,
    x_data,
    save_path=None,
    epoch=None,
    cfm_type=None,
):
    """
    Args:
        trajectories: ODE trajectories (n_steps+1, n_infer, 2)
        cm_onestep: CM one-step samples (n_infer, 2)
        z_samples: initial noise samples (n_infer, 2)
        x_data: training data (n_train, 2)
        cfm_type: CFM coupling type (optional, for title)
    """
    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_infer = trajectories.shape[1]

    ax.scatter(
        x_data[:, 0],
        x_data[:, 1],
        color="black",
        alpha=0.3,
        s=15,
        marker="x",
        label="Real data",
        zorder=1,
    )

    color_ode = "#4472C4"
    color_cm = "#ff7f0e"

    # ODE trajectories
    for i in range(n_infer):
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            color=color_ode,
            alpha=0.4,
            linewidth=1.0,
            zorder=2,
            label="ODE (Euler/Heun)" if i == 0 else None,
        )

    # CM one-step lines
    for i in range(n_infer):
        ax.plot(
            [z_samples[i, 0], cm_onestep[i, 0]],
            [z_samples[i, 1], cm_onestep[i, 1]],
            color=color_cm,
            alpha=0.4,
            linewidth=1.0,
            zorder=2,
            label="CM one-step" if i == 0 else None,
        )

    ax.scatter(
        trajectories[0, :, 0],
        trajectories[0, :, 1],
        color=COLORS["source_x0"],
        alpha=0.8,
        s=30,
        edgecolors="white",
        linewidths=0.5,
        label="z (start)",
        zorder=3,
    )
    ax.scatter(
        trajectories[-1, :, 0],
        trajectories[-1, :, 1],
        color=color_ode,
        alpha=0.8,
        s=30,
        edgecolors="white",
        linewidths=0.5,
        label="x from ODE",
        zorder=3,
    )
    ax.scatter(
        cm_onestep[:, 0],
        cm_onestep[:, 1],
        color=color_cm,
        alpha=0.8,
        s=30,
        edgecolors="white",
        linewidths=0.5,
        label="x from CM (1-step)",
        zorder=3,
        marker="s",
    )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    cfm_str = f' ({cfm_type.upper()})' if cfm_type else ''
    title = f"FACM{cfm_str} (Epoch {epoch})" if epoch is not None else f"FACM{cfm_str}"
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set limits based on data range with padding
    all_x = np.concatenate([x_data[:, 0], trajectories[:, :, 0].flatten(), cm_onestep[:, 0], z_samples[:, 0]])
    all_y = np.concatenate([x_data[:, 1], trajectories[:, :, 1].flatten(), cm_onestep[:, 1], z_samples[:, 1]])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        if epoch is not None:
            print(f"Epoch {epoch} visualization saved to {save_path}")
        else:
            print(f"FACM 2D visualization saved to {save_path}")

    return fig, ax

