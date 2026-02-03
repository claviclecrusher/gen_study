"""
Visualization for BackFlow model
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from visualization.config import COLORS, LABELS, PLOT_PARAMS, setup_plot_style


@torch.no_grad()
def euler_solve(model, z, steps=50):
    """
    Euler method to solve ODE: Integration from t=1 (Noise) to t=0 (Data).
    Boundary Condition v(z, t) ≈ u(z, t, t) is used for velocity.

    This is the EXACT function from original code (lines 284-301).

    Args:
        model: BackFlow model
        z: Initial noise [N, 1]
        steps: Number of Euler steps

    Returns:
        x: Final data point [N, 1]
    """
    B = z.shape[0]
    device = z.device
    dt = -1.0 / steps
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = z

    for i in range(steps):
        t_curr = times[i]
        t_batch = torch.ones(B, device=device) * t_curr
        # Current Velocity Estimation: v(x, t) ≈ u(x, t, t)
        v_pred = model(x, t_batch, t_batch)
        x = x + v_pred * dt

    return x


@torch.no_grad()
def one_step_decode(model, z):
    """
    One-step decode from noise to data.
    This follows the original evaluation code (lines 338-341).

    Args:
        model: BackFlow model
        z: Noise [N, 1]

    Returns:
        x: Decoded data [N, 1]
    """
    B = z.shape[0]
    device = z.device

    # 1-step Decode: r=0, t=1
    r0 = torch.zeros(B, device=device)
    t1 = torch.ones(B, device=device)
    u_dec = model(z, r0, t1)
    x_gen = z - u_dec

    return x_gen


@torch.no_grad()
def one_step_encode(model, x):
    """
    One-step encode from data to noise.
    This follows the original evaluation code (lines 355-356).

    Args:
        model: BackFlow model
        x: Data [N, 1]

    Returns:
        z: Encoded noise [N, 1]
    """
    B = x.shape[0]
    device = x.device

    # 1-step Encode: r=1, t=0
    r1 = torch.ones(B, device=device)
    t0 = torch.zeros(B, device=device)
    u_enc = model(x, r1, t0)
    z_pred = x + u_enc

    return z_pred


@torch.no_grad()
def compute_trajectories(model, z_infer, n_steps=100):
    """
    Compute full trajectories from noise to data using Euler method.

    Args:
        model: BackFlow model
        z_infer: Initial noise points [N, 1]
        n_steps: Number of steps

    Returns:
        trajectories: Array of shape [n_steps+1, N] containing trajectories
    """
    B = z_infer.shape[0]
    device = z_infer.device

    trajectories = []
    x = z_infer.clone()

    dt = -1.0 / n_steps
    times = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

    trajectories.append(x.squeeze().cpu().numpy())

    for i in range(n_steps):
        t_curr = times[i]
        t_batch = torch.ones(B, device=device) * t_curr
        v_pred = model(x, t_batch, t_batch)
        x = x + v_pred * dt
        trajectories.append(x.squeeze().cpu().numpy())

    return np.array(trajectories)


def visualize_training_coupling(z_samples, x_data, coupling_indices, save_path):
    """
    Visualize training data coupling (generated once).

    Args:
        z_samples: Prior samples (training context) [N]
        x_data: Real data (training context) [N]
        coupling_indices: Coupling indices for visualization [N]
        save_path: Path to save figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])

    # Color scheme
    color_coupling = COLORS['coupling_train']  # gray

    # Plot training coupling lines (z -> x) - GRAY
    for i, j in enumerate(coupling_indices):
        ax.plot(
            [1, 0],  # From t=1 (noise) to t=0 (data)
            [z_samples[i], x_data[j]],
            color=color_coupling,
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            zorder=1,
            label='Training coupling' if i == 0 else None
        )

    # Plot source noise points (training)
    ax.scatter(
        np.ones_like(z_samples), z_samples,
        color=COLORS['source_x0'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='z (train, t=1)',
        zorder=3
    )

    # Plot target data points x for reference
    ax.scatter(
        np.zeros_like(x_data), x_data,
        color='black', marker=PLOT_PARAMS['marker_data_x'],
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x (train data, t=0)',
        zorder=3
    )

    # Styling
    all_values = np.concatenate([z_samples, x_data])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('BackFlow Training Coupling')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (data)', 't=1 (noise)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training coupling saved to {save_path}")


def visualize_backflow(trajectories, onestep_final, save_path, epoch=None, x_data=None):
    """
    Visualize BackFlow trajectories (Euler and one-step only, no training coupling).

    Shows:
    1. v: ODE trajectory (blue): Euler ODE with multiple steps
    2. One-step decode (orange): Direct z -> x prediction

    Args:
        trajectories: ODE trajectories [n_steps+1, M] (from t=1 to t=0)
        onestep_final: Final points from 1-step decode [M]
        save_path: Path to save figure
        epoch: Current epoch number (optional, for title)
        x_data: Real data for reference (optional)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])
    n_infer = trajectories.shape[1]
    n_steps = trajectories.shape[0] - 1
    t_space = np.linspace(1, 0, n_steps + 1)  # From noise to data

    # Color scheme
    color_euler = '#4472C4'  # blue (Euler ODE)
    color_onestep = '#ff7f0e'  # orange (one-step)

    # 1. Plot ODE trajectories (Euler) - BLUE
    for i in range(n_infer):
        ax.plot(
            t_space,
            trajectories[:, i],
            color=color_euler,
            alpha=0.7,
            linewidth=PLOT_PARAMS['line_width'] * 1.5,
            zorder=2,
            label='Euler ODE trajectory' if i == 0 else None
        )

    # 2. Plot one-step decode predictions - ORANGE
    z_infer = trajectories[0, :]  # Initial positions (t=1, noise)
    for i in range(n_infer):
        ax.plot(
            [1, 0],  # From t=1 to t=0
            [z_infer[i], onestep_final[i]],
            color=color_onestep,
            alpha=0.7,
            linewidth=PLOT_PARAMS['line_width'] * 2.0,
            zorder=2,
            label='One-step decode' if i == 0 else None
        )

    # 3. Plot points
    x_1_euler = trajectories[-1, :]  # Final points from Euler ODE at t=0 (data)

    # Plot initial noise points
    ax.scatter(
        np.ones(n_infer), z_infer,
        color='gray', marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='z (source, t=1)',
        zorder=3
    )

    # Plot final points from Euler ODE - BLUE
    ax.scatter(
        np.zeros(n_infer), x_1_euler,
        color=color_euler, marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x from Euler',
        zorder=3
    )

    # Plot final points from one-step decode - ORANGE
    ax.scatter(
        np.zeros(n_infer), onestep_final,
        color=color_onestep, marker='s',
        s=PLOT_PARAMS['point_size'] * 1.2,
        alpha=PLOT_PARAMS['point_alpha'],
        label='x from one-step',
        zorder=3
    )

    # Optional: Plot target data points x for reference
    if x_data is not None:
        ax.scatter(
            np.zeros_like(x_data), x_data,
            color='black', marker=PLOT_PARAMS['marker_data_x'],
            s=PLOT_PARAMS['point_size'],
            alpha=PLOT_PARAMS['point_alpha'],
            label='x (train data, t=0)',
            zorder=3
        )

    # Styling
    all_values = [trajectories.flatten(), onestep_final]
    if x_data is not None:
        all_values.append(x_data)
    all_values = np.concatenate(all_values)
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    title = f'BackFlow Trajectory (Epoch {epoch})' if epoch is not None else 'BackFlow Trajectory'
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (data)', 't=1 (noise)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if epoch is not None:
        print(f"Epoch {epoch} visualization saved to {save_path}")
    else:
        print(f"BackFlow visualization saved to {save_path}")
