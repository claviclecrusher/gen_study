"""
Visualization for MeanFlow model
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, setup_plot_style


def visualize_meanflow(z_samples, x_data, trajectories, mean_predictions, coupling_indices, save_path=None, mean_trajectories=None):
    """
    Visualize MeanFlow model training and inference

    Shows four types of mappings:
    1. Training coupling (gray): z -> x straight lines
    2. Instantaneous velocity v (purple): ODE trajectories with multiple steps
    3. Mean velocity u (orange): One-step direct predictions z -> x_pred
    4. Mean velocity u ODE (grapefruit): Multi-step ODE using mean velocity

    Args:
        z_samples: Training source samples z ~ N(0, I) (n_train,)
        x_data: Training target samples x (n_train,)
        trajectories: ODE trajectories from z to x using instantaneous velocity v (n_steps, n_infer)
        mean_predictions: One-step predictions using mean velocity u (n_infer,)
        coupling_indices: Array of indices showing training coupling (n_train,)
        save_path: Path to save the figure (optional)
        mean_trajectories: ODE trajectories using mean velocity u (n_steps, n_infer) (optional)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])
    n_infer = trajectories.shape[1]
    n_steps = trajectories.shape[0] - 1
    t_space = np.linspace(0, 1, n_steps + 1)

    # Color scheme
    color_coupling = COLORS['coupling_train']  # gray
    color_v = '#4472C4'  # blue (instantaneous velocity)
    color_u = '#ff7f0e'  # orange (mean velocity one-step)
    color_u_ode = '#9467bd'  # purple (mean velocity ODE)

    # 1. Plot training coupling lines (z -> x) - GRAY
    for i, j in enumerate(coupling_indices):
        ax.plot(
            [0, 1],
            [z_samples[i], x_data[j]],
            color=color_coupling,
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            zorder=1,
            label='Training coupling' if i == 0 else None
        )

    # 2. Plot ODE trajectories (instantaneous velocity v) - PURPLE
    for i in range(n_infer):
        ax.plot(
            t_space,
            trajectories[:, i],
            color=color_v,
            alpha=0.7,
            linewidth=PLOT_PARAMS['line_width'] * 1.5,
            zorder=2,
            label='v: ODE trajectory' if i == 0 else None
        )

    # 3. Plot mean velocity ODE trajectories (u multi-step) - PURPLE
    if mean_trajectories is not None:
        n_steps_u = mean_trajectories.shape[0] - 1
        t_space_u = np.linspace(0, 1, n_steps_u + 1)
        for i in range(n_infer):
            ax.plot(
                t_space_u,
                mean_trajectories[:, i],
                color=color_u_ode,
                alpha=0.7,
                linewidth=PLOT_PARAMS['line_width'] * 1.5,
                zorder=2,
                label='u: ODE trajectory (multi-step)' if i == 0 else None
            )

    # 4. Plot mean velocity one-step predictions (u) - ORANGE
    z_infer = trajectories[0, :]  # Initial positions
    for i in range(n_infer):
        ax.plot(
            [0, 1],
            [z_infer[i], mean_predictions[i]],
            color=color_u,
            alpha=0.7,
            linewidth=PLOT_PARAMS['line_width'] * 2.0,
            zorder=2,
            label='u: Mean velocity (one-step)' if i == 0 else None
        )

    # 4. Plot points
    x_0 = trajectories[0, :]  # Initial points (inference)
    x_1_v = trajectories[-1, :]  # Final points from ODE (v)
    x_1_u = mean_predictions  # Final points from mean velocity (u)

    # Plot source x0 points (training)
    ax.scatter(
        np.zeros_like(z_samples), z_samples,
        color=COLORS['source_x0'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₀ (train)',
        zorder=3
    )

    # Plot initial points x_0 (inference)
    ax.scatter(
        np.zeros(n_infer), x_0,
        color=COLORS['source_x0_infer'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₀ (infer)',
        zorder=3
    )

    # Plot final points x_1 from ODE trajectory (v) - PURPLE
    ax.scatter(
        np.ones(n_infer), x_1_v,
        color=color_v, marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₁ from v (ODE)',
        zorder=3
    )

    # Plot final points x_1 from mean velocity (u) - ORANGE
    ax.scatter(
        np.ones(n_infer), x_1_u,
        color=color_u, marker='s',
        s=PLOT_PARAMS['point_size'] * 1.2,
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₁ from u (one-step)',
        zorder=3
    )

    # Plot target data points x for reference
    ax.scatter(
        np.ones_like(x_data), x_data,
        color='black', marker=PLOT_PARAMS['marker_data_x'],
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₁ (train data)',
        zorder=3
    )

    # Styling
    all_values = [z_samples, x_data, trajectories.flatten(), mean_predictions]
    if mean_trajectories is not None:
        all_values.append(mean_trajectories.flatten())
    all_values = np.concatenate(all_values)
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('MeanFlow Trajectory')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (x₀)', 't=1 (x₁)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"MeanFlow visualization saved to {save_path}")

    return fig, ax


if __name__ == '__main__':
    print("Testing MeanFlow visualization...")
    n_train, n_infer, n_steps = 50, 10, 100

    # Training data
    z_samples = np.random.randn(n_train)
    x_data = np.random.randn(n_train) * 0.5 + 1.5
    coupling_indices = np.random.permutation(n_train)

    # Inference data
    z_infer_test = np.random.randn(n_infer)

    # ODE trajectories (instantaneous velocity v)
    x_hat_v = z_infer_test + np.random.randn(n_infer) * 0.2 + 1.5
    trajectories = np.zeros((n_steps + 1, n_infer))
    t_space_test = np.linspace(0, 1, n_steps + 1)[:, np.newaxis]
    trajectories = z_infer_test * (1 - t_space_test) + x_hat_v * t_space_test
    trajectories += np.sin(t_space_test * np.pi) * np.random.randn(1, n_infer) * 0.2

    # One-step predictions (mean velocity u)
    mean_predictions = z_infer_test + np.random.randn(n_infer) * 0.3 + 1.4

    fig, ax = visualize_meanflow(
        z_samples, x_data, trajectories, mean_predictions, coupling_indices,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_meanflow.png'
    )
    plt.close()
    print("Test complete!")
