"""
Visualization for MeanFlow model in 2D
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


def visualize_meanflow_2d(trajectories, mean_onestep, z_samples, x_data, save_path=None, epoch=None, cfm_type=None):
    """
    Visualize MeanFlow model in 2D with trajectories

    Args:
        trajectories: ODE trajectories with instantaneous velocity v (n_steps+1, n_infer, 2)
        mean_onestep: One-step predictions with mean velocity u (n_infer, 2)
        z_samples: Initial noise samples (n_infer, 2)
        x_data: Training target samples x (n_train, 2)
        save_path: Path to save the figure (optional)
        epoch: Current epoch number (optional, for title)
        cfm_type: CFM coupling type (optional, for title)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_infer = trajectories.shape[1]

    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
              color='black', alpha=0.3, s=15, marker='x',
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

    # Compute difference between ODE solver and u one-step
    ode_final = trajectories[-1, :, :]  # [n_infer, 2] - ODE final results
    diff = ode_final - mean_onestep  # [n_infer, 2] - difference vector
    diff_norm = np.linalg.norm(diff, axis=1)  # [n_infer] - L2 norm of difference
    diff_mean = np.mean(diff_norm)
    diff_var = np.var(diff_norm)
    diff_std = np.std(diff_norm)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    cfm_str = f' ({cfm_type.upper()})' if cfm_type else ''
    title = f'MeanFlow{cfm_str} (Epoch {epoch})' if epoch is not None else f'MeanFlow{cfm_str}'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set limits based on data range with padding
    all_x = np.concatenate([x_data[:, 0], trajectories[:, :, 0].flatten(), mean_onestep[:, 0], z_samples[:, 0]])
    all_y = np.concatenate([x_data[:, 1], trajectories[:, :, 1].flatten(), mean_onestep[:, 1], z_samples[:, 1]])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect('equal', adjustable='box')
    
    # Add statistics text
    stats_text = f'||ODE - u one-step||:\nMean={diff_mean:.4f}, Std={diff_std:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
