"""
Visualization for ModeFlowMatching (ModeFM) model in 2D

Dedicated visualization module for ModeFM 2D. Same structure as FM 2D for now;
extensible for future modefm-specific visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


def visualize_modefm_2d(trajectories, x_data, save_path=None, epoch=None, cfm_type=None):
    """
    Visualize ModeFlowMatching model in 2D with trajectories

    Args:
        trajectories: ODE trajectories from z to x (n_steps+1, n_infer, 2)
        x_data: Training target samples x (n_train, 2)
        save_path: Path to save the figure (optional)
        epoch: Current epoch number (optional, for title)
        cfm_type: CFM coupling type (optional, for title)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_steps = trajectories.shape[0]
    n_infer = trajectories.shape[1]

    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
               color='black', alpha=0.3, s=15, edgecolors='none', marker='x',
               label='Real data', zorder=1)

    # Plot trajectories (z -> x)
    for i in range(n_infer):
        ax.plot(trajectories[:, i, 0], trajectories[:, i, 1],
               color=COLORS['trajectory'], alpha=0.5, linewidth=1.0, zorder=2)

    # Plot starting points (z)
    ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1],
               color=COLORS['source_x0'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='z (start)', zorder=3)

    # Plot final points (generated x)
    ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1],
               color=COLORS['infer_x_hat'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='x (generated)', zorder=3)

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    cfm_str = f' ({cfm_type.upper()})' if cfm_type else ''
    title = f'ModeFM{cfm_str} (Epoch {epoch})' if epoch is not None else f'ModeFM{cfm_str}'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set limits based on data range with padding
    all_x = np.concatenate([x_data[:, 0], trajectories[:, :, 0].flatten()])
    all_y = np.concatenate([x_data[:, 1], trajectories[:, :, 1].flatten()])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None:
            print(f"Epoch {epoch} visualization saved to {save_path}")
        else:
            print(f"ModeFM 2D visualization saved to {save_path}")

    return fig, ax
