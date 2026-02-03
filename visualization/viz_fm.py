"""
Visualization for Flow Matching model
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, setup_plot_style


def visualize_fm(z_samples, x_data, trajectories, coupling_indices, save_path=None, vector_info=None):
    """
    Visualize Flow Matching model training and inference

    Args:
        z_samples: Training source samples z ~ N(0, I) (n_train,)
        x_data: Training target samples x (n_train,)
        trajectories: ODE trajectories from z to x_hat (n_steps, n_infer)
        coupling_indices: Array of indices showing training coupling (n_train,)
        save_path: Path to save the figure (optional)
        vector_info: Dictionary containing velocity and score vectors at specific time points (optional)
                     Keys: 't_points', 'x_t_samples', 'velocities', 'scores'
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])
    n_infer = trajectories.shape[1]
    n_steps = trajectories.shape[0] - 1
    t_space = np.linspace(0, 1, n_steps + 1)

    # 1. Plot training coupling lines (z -> x)
    for i, j in enumerate(coupling_indices):
        ax.plot(
            [0, 1],
            [z_samples[i], x_data[j]],
            color=COLORS['coupling_train'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            zorder=1,
            label=LABELS['coupling_train'] if i == 0 else None
        )

    # 2. Plot ODE trajectories (z -> x_1)
    for i in range(n_infer):
        ax.plot(
            t_space,
            trajectories[:, i],
            color=COLORS['trajectory'],
            alpha=0.7,
            linewidth=PLOT_PARAMS['line_width'] * 1.5,
            zorder=2,
            label=LABELS['trajectory'] if i == 0 else None
        )

    # 3. Plot points
    x_0 = trajectories[0, :]
    x_1 = trajectories[-1, :]

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

    # Plot final points x_1 (inference)
    ax.scatter(
        np.ones(n_infer), x_1,
        color=COLORS['infer_x_hat'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₁ (infer)',
        zorder=3
    )
    
    # Plot target data points x for reference
    ax.scatter(
        np.ones_like(x_data), x_data,
        color=COLORS['data_x'], marker=PLOT_PARAMS['marker_data_x'],
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x₁ (train data)',
        zorder=3
    )

    # Plot velocity and score vectors if provided
    if vector_info is not None:
        t_points = vector_info['t_points']
        x_t_samples = vector_info['x_t_samples']
        velocities = vector_info['velocities']
        scores = vector_info['scores']

        # Compute scaling factors to make arrows visible and similar in size
        # Velocity is dx/dt, so we visualize it as (dt, velocity*dt)
        # Score is gradient in x direction, visualize as (0, score*scale)
        dt = 0.08  # time step for visualization

        # Normalize velocities and scores to similar scale
        v_max = np.abs(velocities).max() if len(velocities) > 0 else 1.0
        s_max = np.abs(scores).max() if len(scores) > 0 else 1.0
        max_scale = max(v_max, s_max)

        # Scale factor for arrows (adjust to make them visible but not too large)
        arrow_scale = 0.5  # Adjust this to control overall arrow size

        # Plot velocity arrows (red)
        for i, (t, x_t, v) in enumerate(zip(t_points, x_t_samples, velocities)):
            # Velocity arrow: direction (dt, v*dt) normalized
            dx_arrow = dt
            dy_arrow = v * dt * arrow_scale / max_scale

            ax.arrow(
                t, x_t, dx_arrow, dy_arrow,
                head_width=0.02, head_length=0.015,
                fc='red', ec='red',
                alpha=0.8, linewidth=1.5,
                zorder=4,
                label='Velocity' if i == 0 else None
            )

        # Plot score arrows (blue)
        for i, (t, x_t, s) in enumerate(zip(t_points, x_t_samples, scores)):
            # Score arrow: direction (0, s*scale) normalized to similar size as velocity
            dx_arrow = 0
            dy_arrow = s * dt * arrow_scale / max_scale

            ax.arrow(
                t, x_t, dx_arrow, dy_arrow,
                head_width=0.02, head_length=0.015,
                fc='blue', ec='blue',
                alpha=0.8, linewidth=1.5,
                zorder=4,
                label='Score' if i == 0 else None
            )

        # Plot the sample points where vectors are computed
        ax.scatter(
            t_points, x_t_samples,
            color='black', marker='x',
            s=PLOT_PARAMS['point_size'] * 1.5,
            linewidths=2,
            alpha=0.9,
            label='Vector points',
            zorder=5
        )

    # Styling
    all_values = np.concatenate([z_samples, x_data, trajectories.flatten()])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.set_title('Flow Matching: x₀ → x₁ Trajectories')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (x₀)', 't=1 (x₁)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Flow Matching visualization saved to {save_path}")

    return fig, ax


if __name__ == '__main__':
    print("Testing Flow Matching visualization...")
    n_train, n_infer, n_steps = 50, 10, 100
    z_samples = np.random.randn(n_train)
    x_data = np.random.randn(n_train) * 0.5 + 1.5
    z_infer_test = np.random.randn(n_infer)
    x_hat_test = z_infer_test + np.random.randn(n_infer) * 0.2
    trajectories = np.zeros((n_steps + 1, n_infer))
    t_space_test = np.linspace(0, 1, n_steps + 1)[:, np.newaxis]
    trajectories = z_infer_test * (1 - t_space_test) + x_hat_test * t_space_test
    trajectories += np.sin(t_space_test * np.pi) * np.random.randn(1, n_infer) * 0.2
    coupling_indices = np.random.permutation(n_train)
    fig, ax = visualize_fm(
        z_samples, x_data, trajectories, coupling_indices,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_fm.png'
    )
    plt.close()
    print("Test complete!")
