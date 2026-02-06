"""
Visualization for TopK-OTCFM model
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, setup_plot_style


def visualize_topk_fm(z_samples, x_data, trajectories, coupling_indices, save_path=None, epoch=None, is_pretraining=False):
    """
    Visualize TopK-OTCFM model training and inference
    
    Args:
        z_samples: Training source samples z ~ N(0, I) (n_train,)
        x_data: Training target samples x (n_train,)
        trajectories: ODE trajectories from z to x_hat (n_steps, n_infer)
        coupling_indices: Array of indices showing training coupling (n_train,)
        save_path: Path to save the figure (optional)
        epoch: Current epoch number (optional, for title)
        is_pretraining: Whether currently in pretraining stage (optional)
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
    
    # Styling
    all_values = np.concatenate([z_samples, x_data, trajectories.flatten()])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])
    
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    stage = "Pretraining" if is_pretraining else "Retraining"
    title = f'TopK-OTCFM: {stage} (Epoch {epoch})' if epoch is not None else f'TopK-OTCFM: {stage}'
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['t=0 (x₀)', 't=1 (x₁)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None:
            print(f"Epoch {epoch} visualization saved to {save_path}")
        else:
            print(f"TopK-OTCFM visualization saved to {save_path}")
    
    return fig, ax


if __name__ == '__main__':
    print("Testing TopK-OTCFM visualization...")
    from data.synthetic import generate_data, sample_prior
    
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
    
    fig, ax = visualize_topk_fm(
        z_samples, x_data, trajectories, coupling_indices,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_topk_fm.png',
        epoch=1,
        is_pretraining=False
    )
    plt.close()
    print("Test complete!")
