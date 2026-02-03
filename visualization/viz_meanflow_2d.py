"""
Visualization for MeanFlow model in 2D
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


def visualize_meanflow_2d(trajectories, mean_onestep, z_samples, x_data, save_path=None, epoch=None):
    """
    Visualize MeanFlow model in 2D with trajectories

    Args:
        trajectories: ODE trajectories with instantaneous velocity v (n_steps+1, n_infer, 2)
        mean_onestep: One-step predictions with mean velocity u (n_infer, 2)
        z_samples: Initial noise samples (n_infer, 2)
        x_data: Training target samples x (n_train, 2)
        save_path: Path to save the figure (optional)
        epoch: Current epoch number (optional, for title)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_infer = trajectories.shape[1]

    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
              color='black', alpha=0.3, s=15, edgecolors='none', marker='x',
              label='Real data', zorder=1)

    # Plot ODE trajectories (instantaneous velocity v) - blue
    for i in range(n_infer):
        ax.plot(trajectories[:, i, 0], trajectories[:, i, 1],
               color='#4472C4', alpha=0.4, linewidth=1.0, zorder=2,
               label='ODE (v)' if i == 0 else None)

    # Plot one-step trajectories (mean velocity u) - orange straight lines
    for i in range(n_infer):
        ax.plot([z_samples[i, 0], mean_onestep[i, 0]],
               [z_samples[i, 1], mean_onestep[i, 1]],
               color='#ff7f0e', alpha=0.4, linewidth=1.0, zorder=2,
               label='One-step (u)' if i == 0 else None)

    # Plot starting points (z)
    ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1],
              color=COLORS['source_x0'], alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='z (start)', zorder=3)

    # Plot final points from ODE (v)
    ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1],
              color='#4472C4', alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='x from ODE', zorder=3)

    # Plot final points from one-step (u)
    ax.scatter(mean_onestep[:, 0], mean_onestep[:, 1],
              color='#ff7f0e', alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='x from one-step', zorder=3, marker='s')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    title = f'MeanFlow (Epoch {epoch})' if epoch is not None else 'MeanFlow'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None:
            print(f"Epoch {epoch} visualization saved to {save_path}")
        else:
            print(f"MeanFlow 2D visualization saved to {save_path}")

    return fig, ax


if __name__ == '__main__':
    print("Testing MeanFlow 2D visualization...")
    from data.synthetic import generate_data_2d, sample_prior

    n_train, n_infer = 500, 20
    n_steps = 50
    x_data = generate_data_2d(n_samples=n_train, seed=42)

    # Create fake trajectories for testing
    z_init = sample_prior(n_samples=n_infer, seed=43, dim=2)
    x_final_ode = z_init + np.array([1.5, 1.5])
    x_final_onestep = z_init + np.array([1.7, 1.7])

    # Linear interpolation for ODE trajectories
    trajectories = np.zeros((n_steps + 1, n_infer, 2))
    for i in range(n_steps + 1):
        t = i / n_steps
        trajectories[i] = (1 - t) * z_init + t * x_final_ode

    visualize_meanflow_2d(
        trajectories=trajectories,
        mean_onestep=x_final_onestep,
        z_samples=z_init,
        x_data=x_data,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_meanflow_2d.png',
        epoch=1
    )
    print("Test complete!")
