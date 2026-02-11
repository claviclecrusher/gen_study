"""
Visualization for ModeFlowMatching (ModeFM) model

Dedicated visualization module for ModeFM. Same structure as FM for now;
extensible for future modefm-specific visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, setup_plot_style


def visualize_modefm(z_samples, x_data, trajectories, coupling_indices, save_path=None, vector_info=None):
    """
    Visualize ModeFlowMatching model training and inference

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

        dt = 0.08
        v_max = np.abs(velocities).max() if len(velocities) > 0 else 1.0
        s_max = np.abs(scores).max() if len(scores) > 0 else 1.0
        max_scale = max(v_max, s_max)
        arrow_scale = 0.5

        for i, (t, x_t, v) in enumerate(zip(t_points, x_t_samples, velocities)):
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

        for i, (t, x_t, s) in enumerate(zip(t_points, x_t_samples, scores)):
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
    ax.set_title('ModeFM: x₀ → x₁ Trajectories (Gaussian Kernel Loss)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (x₀)', 't=1 (x₁)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ModeFM visualization saved to {save_path}")

    return fig, ax
