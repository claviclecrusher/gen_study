"""
Visualization for BackFlow model in 2D
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


@torch.no_grad()
def compute_trajectories_2d(model, z, steps=50):
    """
    Compute full trajectories using Euler method for 2D: Integration from t=1 (Noise) to t=0 (Data).

    Args:
        model: BackFlow model
        z: Initial noise [N, 2]
        steps: Number of Euler steps

    Returns:
        trajectories: Full trajectory [steps+1, N, 2]
    """
    B = z.shape[0]
    device = z.device
    dt = -1.0 / steps
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)

    # Store trajectories
    trajectories = []
    x = z.clone()
    trajectories.append(x.cpu().numpy())

    for i in range(steps):
        t_curr = times[i]
        t_batch = torch.ones(B, device=device) * t_curr
        v_pred = model(x, t_batch, t_batch)
        x = x + v_pred * dt
        trajectories.append(x.cpu().numpy())

    return np.array(trajectories)  # (steps+1, N, 2)


@torch.no_grad()
def one_step_decode_2d(model, z):
    """
    One-step decode from noise to data for 2D.

    Args:
        model: BackFlow model
        z: Noise [N, 2]

    Returns:
        x: Decoded data [N, 2]
    """
    B = z.shape[0]
    device = z.device

    r0 = torch.zeros(B, device=device)
    t1 = torch.ones(B, device=device)
    u_dec = model(z, r0, t1)
    x_gen = z - u_dec

    return x_gen


def visualize_backflow_2d(trajectories, onestep_samples, x_data, save_path=None, epoch=None, cfm_type=None):
    """
    Visualize BackFlow model in 2D with trajectories

    Args:
        trajectories: Euler ODE trajectories from z to x (n_steps+1, n_infer, 2)
        onestep_samples: One-step decoded samples (n_infer, 2)
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
              color='black', alpha=0.3, s=15, edgecolors='none', marker='x',
              label='Real data', zorder=1)

    # Plot Euler ODE trajectories - blue
    for i in range(n_infer):
        ax.plot(trajectories[:, i, 0], trajectories[:, i, 1],
               color='#4472C4', alpha=0.5, linewidth=1.0, zorder=2,
               label='Euler ODE' if i == 0 else None)

    # Plot one-step trajectories (straight lines from z to decoded point) - orange
    for i in range(n_infer):
        ax.plot([trajectories[0, i, 0], onestep_samples[i, 0]],
               [trajectories[0, i, 1], onestep_samples[i, 1]],
               color='#ff7f0e', alpha=0.5, linewidth=1.0, zorder=2,
               label='One-step' if i == 0 else None)

    # Plot starting points (z at t=1)
    ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1],
              color='gray', alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='z (start, t=1)', zorder=3)

    # Plot final points from Euler ODE (x at t=0)
    ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1],
              color='#4472C4', alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='x from Euler', zorder=3)

    # Plot one-step decoded samples - orange
    ax.scatter(onestep_samples[:, 0], onestep_samples[:, 1],
              color='#ff7f0e', alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='x from one-step', zorder=3, marker='s')

    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    cfm_str = f' ({cfm_type.upper()})' if cfm_type else ''
    title = f'BackFlow{cfm_str} (Epoch {epoch})' if epoch is not None else f'BackFlow{cfm_str}'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set limits based on data range with padding
    all_x = np.concatenate([x_data[:, 0], trajectories[:, :, 0].flatten(), onestep_samples[:, 0]])
    all_y = np.concatenate([x_data[:, 1], trajectories[:, :, 1].flatten(), onestep_samples[:, 1]])
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
            print(f"BackFlow 2D visualization saved to {save_path}")

    return fig, ax


if __name__ == '__main__':
    print("Testing BackFlow 2D visualization...")
    from data.synthetic import generate_data_2d, sample_prior

    n_train, n_infer = 500, 20
    n_steps = 50
    x_data = generate_data_2d(n_samples=n_train, seed=42)

    # Create fake trajectories for testing
    z_init = sample_prior(n_samples=n_infer, seed=43, dim=2)
    x_final_ode = z_init + np.array([-1.5, -1.5])
    x_final_onestep = z_init + np.array([-1.7, -1.7])

    # Linear interpolation for trajectories (reversed: t=1 to t=0)
    trajectories = np.zeros((n_steps + 1, n_infer, 2))
    for i in range(n_steps + 1):
        t = i / n_steps  # 0 to 1
        # t=1 (noise) -> t=0 (data)
        trajectories[i] = (1 - t) * z_init + t * x_final_ode

    visualize_backflow_2d(
        trajectories=trajectories,
        onestep_samples=x_final_onestep,
        x_data=x_data,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_backflow_2d.png',
        epoch=1
    )
    print("Test complete!")
