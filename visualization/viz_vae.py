"""
Visualization for Variational Autoencoder (VAE)
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, add_epsilon_noise, setup_plot_style


def visualize_vae(z_prior, x_data, z_hat, x_hat, z_infer, x_infer, save_path=None, beta=1.0):
    """
    Visualize VAE training and inference

    Args:
        z_prior: Prior samples z ~ N(0, I) (n_prior,)
        x_data: Data samples x (n_data,)
        z_hat: Encoded latent features ẑ from x_data (n_data,)
        x_hat: Decoding outputs D(E(x)) from z_hat (n_data,)
        z_infer: Inference input z samples for D(z) (n_infer,)
        x_infer: Inference outputs D(z) from z_infer (n_infer,)
        save_path: Path to save the figure (optional)
        beta: KL divergence weight (default: 1.0)

    Note:
        In VAE, z_hat should be closer to z_prior due to KL regularization
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])

    # Add epsilon noise and type-specific shifts for x-axis separation
    z_prior_x_coords = np.full_like(z_prior, PLOT_PARAMS['left_x'] + PLOT_PARAMS['shift_prior_z']) + add_epsilon_noise(z_prior)
    z_hat_x_coords = np.full_like(z_hat, PLOT_PARAMS['left_x'] + PLOT_PARAMS['shift_latent_z_hat']) + add_epsilon_noise(z_hat)
    z_infer_x_coords = np.full_like(z_infer, PLOT_PARAMS['left_x']) + add_epsilon_noise(z_infer)
    x_data_x_coords = np.full_like(x_data, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_data_x']) + add_epsilon_noise(x_data)
    x_hat_x_coords = np.full_like(x_hat, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_output_x_hat']) + add_epsilon_noise(x_hat)
    x_infer_x_coords = np.full_like(x_infer, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_infer_x_hat']) + add_epsilon_noise(x_infer)

    # Plot encoding lines (x → ẑ)
    for i in range(len(x_data)):
        ax.plot(
            [x_data_x_coords[i], z_hat_x_coords[i]],
            [x_data[i], z_hat[i]],
            color=COLORS['encode_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            linestyle='--',
            zorder=1
        )

    # Plot decoding lines (ẑ → x̂)
    for i in range(len(z_hat)):
        ax.plot(
            [z_hat_x_coords[i], x_hat_x_coords[i]],
            [z_hat[i], x_hat[i]],
            color=COLORS['decode_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            linestyle='--',
            zorder=2
        )

    # Plot inference lines (z → D(z))
    for i in range(len(z_infer)):
        ax.plot(
            [z_infer_x_coords[i], x_infer_x_coords[i]],
            [z_infer[i], x_infer[i]],
            color=COLORS['infer_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            zorder=2
        )

    # Plot points in random order to avoid systematic occlusion
    # Combine all points with their properties
    all_points = []

    # Left panel: prior z
    for i in range(len(z_prior)):
        all_points.append({
            'x': z_prior_x_coords[i], 'y': z_prior[i],
            'color': COLORS['prior_z'], 'marker': 'o',
            'label': LABELS['prior_z'] if i == 0 else None
        })

    # Left panel: encoded ẑ
    for i in range(len(z_hat)):
        all_points.append({
            'x': z_hat_x_coords[i], 'y': z_hat[i],
            'color': COLORS['latent_z_hat'], 'marker': 'o',
            'label': LABELS['latent_z_hat'] if i == 0 else None
        })

    # Right panel: data x
    for i in range(len(x_data)):
        all_points.append({
            'x': x_data_x_coords[i], 'y': x_data[i],
            'color': COLORS['data_x'], 'marker': PLOT_PARAMS['marker_data_x'],
            'label': LABELS['data_x'] if i == 0 else None
        })

    # Right panel: decoding output D(E(x))
    for i in range(len(x_hat)):
        all_points.append({
            'x': x_hat_x_coords[i], 'y': x_hat[i],
            'color': COLORS['output_x_hat'], 'marker': 'o',
            'label': LABELS['output_x_hat'] if i == 0 else None
        })

    # Right panel: inference output D(z)
    for i in range(len(x_infer)):
        all_points.append({
            'x': x_infer_x_coords[i], 'y': x_infer[i],
            'color': COLORS['infer_x_hat'], 'marker': 'o',
            'label': LABELS['infer_x_hat'] if i == 0 else None
        })

    # Shuffle points randomly
    np.random.shuffle(all_points)

    # Plot points in random order
    legend_labels = {}
    for point in all_points:
        if point['label'] and point['label'] not in legend_labels:
            # First point of each type - add to legend
            ax.scatter(
                point['x'], point['y'],
                c=point['color'], marker=point['marker'],
                s=PLOT_PARAMS['point_size'],
                alpha=PLOT_PARAMS['point_alpha'],
                label=point['label'],
                zorder=3
            )
            legend_labels[point['label']] = True
        else:
            # Other points - no label
            ax.scatter(
                point['x'], point['y'],
                c=point['color'], marker=point['marker'],
                s=PLOT_PARAMS['point_size'],
                alpha=PLOT_PARAMS['point_alpha'],
                zorder=3
            )

    # Create custom legend entries for lines
    from matplotlib.lines import Line2D
    line_encode = Line2D([0], [0], color=COLORS['encode_line'],
                        linewidth=2, alpha=0.7, label=LABELS['encode_line'])
    line_decode = Line2D([0], [0], color=COLORS['decode_line'],
                        linewidth=2, alpha=0.7, label=LABELS['decode_line'])
    line_infer = Line2D([0], [0], color=COLORS['infer_line'],
                        linewidth=2, alpha=0.7, label=LABELS['infer_line'])

    # Get current handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([line_encode, line_decode, line_infer])
    labels.extend([LABELS['encode_line'], LABELS['decode_line'], LABELS['infer_line']])

    # Styling - reduce empty space on sides
    ax.set_xlim(-0.15, 1.15)
    all_values = np.concatenate([z_prior, x_data, z_hat, x_hat, z_infer, x_infer])
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Space')
    ax.set_ylabel('Value')
    ax.set_title(f'VAE: Encoding and Decoding with KL Regularization (β={beta})')
    ax.set_xticks([PLOT_PARAMS['left_x'], PLOT_PARAMS['right_x']])
    ax.set_xticklabels(['Latent (z, ẑ)', 'Data (x, x̂)'])
    ax.legend(handles=handles, labels=labels, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"VAE visualization saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing VAE visualization...")

    n_samples = 100

    # Generate dummy data
    z_prior = np.random.randn(n_samples)
    x_data = np.random.randn(n_samples) * 0.5 + 1.5

    # Simulate VAE behavior
    # Encoded latent should be closer to prior than AE due to KL regularization
    z_hat = x_data * 0.3 + np.random.randn(n_samples) * 0.5

    # Reconstructed output might be less accurate than AE due to regularization
    x_hat = x_data + np.random.randn(n_samples) * 0.3

    # Inference samples
    n_infer = 50
    z_infer = np.random.randn(n_infer)
    x_infer = z_infer * 0.3 + 1.5 + np.random.randn(n_infer) * 0.4

    fig, ax = visualize_vae(
        z_prior, x_data, z_hat, x_hat, z_infer, x_infer,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_vae.png'
    )
    plt.close()
    print("Test complete!")
