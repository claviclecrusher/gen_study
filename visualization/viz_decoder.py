"""
Visualization for Non-identifiable Decoder
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, add_epsilon_noise, setup_plot_style


def visualize_decoder(z_samples, x_data, x_hat, z_infer, coupling_indices, save_path=None):
    """
    Visualize Non-identifiable Decoder training and inference
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])

    z_x = PLOT_PARAMS['left_x']
    x_x = PLOT_PARAMS['right_x']

    # Add epsilon noise for x-axis separation
    z_x_coords = np.full_like(z_samples, z_x) + add_epsilon_noise(z_samples)
    x_x_coords = np.full_like(x_data, x_x) + add_epsilon_noise(x_data)
    x_hat_x_coords = np.full_like(x_hat, x_x) + add_epsilon_noise(x_hat, scale=PLOT_PARAMS['epsilon_scale'] * 2)
    z_infer_x_coords = np.full_like(z_infer, z_x) + add_epsilon_noise(z_infer, scale=PLOT_PARAMS['epsilon_scale'] * 2)

    # Plot training coupling lines
    for i, j in enumerate(coupling_indices):
        ax.plot([z_x_coords[i], x_x_coords[j]], [z_samples[i], x_data[j]],
                color=COLORS['coupling_train'], alpha=PLOT_PARAMS['line_alpha'],
                linewidth=PLOT_PARAMS['line_width'], zorder=1,
                label=LABELS['coupling_train'] if i == 0 else None)

    # Plot inference coupling lines
    for i in range(len(z_infer)):
        ax.plot([z_infer_x_coords[i], x_hat_x_coords[i]], [z_infer[i], x_hat[i]],
                color=COLORS['coupling_infer'], alpha=PLOT_PARAMS['line_alpha'],
                linewidth=PLOT_PARAMS['line_width'], zorder=2,
                label=LABELS['coupling_infer'] if i == 0 else None)

    # Plot points
    ax.scatter(z_x_coords, z_samples,
               c=COLORS['prior_z'], marker='o', s=PLOT_PARAMS['point_size'],
               alpha=PLOT_PARAMS['point_alpha'], label=LABELS['source_z'], zorder=3)
    
    ax.scatter(z_infer_x_coords, z_infer,
               c=COLORS['prior_z_infer'], marker='o', s=PLOT_PARAMS['point_size'],
               alpha=PLOT_PARAMS['point_alpha'], label=LABELS['prior_z'] + ' (infer)', zorder=3)

    ax.scatter(x_x_coords, x_data,
               c=COLORS['data_x'], marker=PLOT_PARAMS['marker_data_x'], s=PLOT_PARAMS['point_size'],
               alpha=PLOT_PARAMS['point_alpha'], label=LABELS['data_x'], zorder=3)

    ax.scatter(x_hat_x_coords, x_hat,
               c=COLORS['infer_x_hat'], marker='o', s=PLOT_PARAMS['point_size'],
               alpha=PLOT_PARAMS['point_alpha'], label='Inference D(z)', zorder=3)

    # Styling
    ax.set_xlim(z_x - 0.1, x_x + 0.1)
    all_values = np.concatenate([z_samples, x_data, x_hat, z_infer])
    ax.set_ylim(all_values.min() - PLOT_PARAMS['y_padding'], all_values.max() + PLOT_PARAMS['y_padding'])
    
    ax.set_xlabel('Space')
    ax.set_ylabel('Value')
    ax.set_title('Non-identifiable Decoder: Training and Inference')
    ax.set_xticks([z_x, x_x])
    ax.set_xticklabels(['Source (z)', 'Target (x, D(z))'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Decoder visualization saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    print("Testing decoder visualization...")
    n_train, n_infer = 100, 50
    z_samples = np.random.randn(n_train)
    x_data = np.random.randn(n_train) * 0.5 + 1.5
    x_hat = np.ones(n_infer) * x_data.mean()
    z_infer = np.random.randn(n_infer)
    coupling_indices = np.random.permutation(n_train)
    fig, ax = visualize_decoder(
        z_samples, x_data, x_hat, z_infer, coupling_indices,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_decoder.png'
    )
    plt.close()
    print("Test complete!")
