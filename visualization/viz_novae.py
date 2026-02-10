"""
Visualization for Noise Oriented VAE (NO-VAE) - 1D

Provides two visualization functions:
1. visualize_novae() - Detailed standalone visualization (for main.py)
   Shows prior z, encoder z_, data x, reconstruction, and inference.
2. visualize_novae_1d() - Comparison format (for run_comparison.py)
   Shows z -> D(z) one-step mapping, similar to flow model trajectory format.
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, LABELS, PLOT_PARAMS, add_epsilon_noise, setup_plot_style


def visualize_novae(z_prior, x_data, z_, x_hat, z_infer, x_infer,
                   save_path=None, temperature=0.1):
    """
    Detailed NO-VAE visualization showing full encoder-decoder pipeline (1D).

    Layout similar to VAE visualization:
    - Left panel (latent space): prior z, encoder output z_
    - Right panel (data space): data x, reconstruction x_hat, inference D(z)
    - Lines: encoding (x -> z_), decoding (z_ -> x_hat), inference (z -> D(z))

    Args:
        z_prior: Prior samples z ~ N(0, I) (n_prior,)
        x_data: Data samples x (n_data,)
        z_: Encoder output from x_data (n_data,)
        x_hat: Reconstruction D(softNN(z_, prior)) from x_data (n_data,)
        z_infer: Inference input z samples for D(z) (n_infer,)
        x_infer: Inference outputs D(z) from z_infer (n_infer,)
        save_path: Path to save the figure (optional)
        temperature: Soft NN temperature (for title display)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])

    # Add epsilon noise for x-axis separation
    z_prior_x = np.full_like(z_prior, PLOT_PARAMS['left_x'] + PLOT_PARAMS['shift_prior_z']) + add_epsilon_noise(z_prior)
    z_enc_x = np.full_like(z_, PLOT_PARAMS['left_x'] + PLOT_PARAMS['shift_latent_z_hat']) + add_epsilon_noise(z_)
    z_infer_x = np.full_like(z_infer, PLOT_PARAMS['left_x']) + add_epsilon_noise(z_infer)
    x_data_x = np.full_like(x_data, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_data_x']) + add_epsilon_noise(x_data)
    x_hat_x = np.full_like(x_hat, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_output_x_hat']) + add_epsilon_noise(x_hat)
    x_infer_x = np.full_like(x_infer, PLOT_PARAMS['right_x'] + PLOT_PARAMS['shift_infer_x_hat']) + add_epsilon_noise(x_infer)

    # Plot encoding lines (x -> z_)
    for i in range(len(x_data)):
        ax.plot(
            [x_data_x[i], z_enc_x[i]],
            [x_data[i], z_[i]],
            color=COLORS['encode_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            linestyle='--',
            zorder=1
        )

    # Plot decoding lines (z_ -> x_hat) - through soft NN
    for i in range(len(z_)):
        ax.plot(
            [z_enc_x[i], x_hat_x[i]],
            [z_[i], x_hat[i]],
            color=COLORS['decode_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            linestyle='--',
            zorder=2
        )

    # Plot inference lines (z -> D(z))
    for i in range(len(z_infer)):
        ax.plot(
            [z_infer_x[i], x_infer_x[i]],
            [z_infer[i], x_infer[i]],
            color=COLORS['infer_line'],
            alpha=PLOT_PARAMS['line_alpha'],
            linewidth=PLOT_PARAMS['line_width'],
            zorder=2
        )

    # Collect all points for random-order plotting (avoid systematic occlusion)
    all_points = []

    # Prior z
    for i in range(len(z_prior)):
        all_points.append({
            'x': z_prior_x[i], 'y': z_prior[i],
            'color': COLORS['prior_z'], 'marker': 'o',
            'label': LABELS['prior_z'] if i == 0 else None
        })

    # Encoder output z_
    for i in range(len(z_)):
        all_points.append({
            'x': z_enc_x[i], 'y': z_[i],
            'color': COLORS['latent_z_hat'], 'marker': 'o',
            'label': 'Encoder z\u0305' if i == 0 else None
        })

    # Data x
    for i in range(len(x_data)):
        all_points.append({
            'x': x_data_x[i], 'y': x_data[i],
            'color': COLORS['data_x'], 'marker': PLOT_PARAMS['marker_data_x'],
            'label': LABELS['data_x'] if i == 0 else None
        })

    # Reconstruction x_hat
    for i in range(len(x_hat)):
        all_points.append({
            'x': x_hat_x[i], 'y': x_hat[i],
            'color': COLORS['output_x_hat'], 'marker': 'o',
            'label': 'Recon D(z_sel)' if i == 0 else None
        })

    # Inference output D(z)
    for i in range(len(x_infer)):
        all_points.append({
            'x': x_infer_x[i], 'y': x_infer[i],
            'color': COLORS['infer_x_hat'], 'marker': 'o',
            'label': LABELS['infer_x_hat'] if i == 0 else None
        })

    # Shuffle and plot
    np.random.shuffle(all_points)
    legend_labels = {}
    for point in all_points:
        label = point['label']
        if label and label not in legend_labels:
            ax.scatter(
                point['x'], point['y'],
                c=point['color'], marker=point['marker'],
                s=PLOT_PARAMS['point_size'],
                alpha=PLOT_PARAMS['point_alpha'],
                label=label, zorder=3
            )
            legend_labels[label] = True
        else:
            ax.scatter(
                point['x'], point['y'],
                c=point['color'], marker=point['marker'],
                s=PLOT_PARAMS['point_size'],
                alpha=PLOT_PARAMS['point_alpha'],
                zorder=3
            )

    # Line legend entries
    from matplotlib.lines import Line2D
    line_encode = Line2D([0], [0], color=COLORS['encode_line'],
                         linewidth=2, alpha=0.7, linestyle='--',
                         label='Encode (x \u2192 z\u0305)')
    line_decode = Line2D([0], [0], color=COLORS['decode_line'],
                         linewidth=2, alpha=0.7, linestyle='--',
                         label='Decode (z\u0305 \u2192 x\u0302)')
    line_infer = Line2D([0], [0], color=COLORS['infer_line'],
                        linewidth=2, alpha=0.7,
                        label='Inference (z \u2192 D(z))')

    handles, labels = ax.get_legend_handles_labels()
    handles.extend([line_encode, line_decode, line_infer])
    labels.extend(['Encode (x \u2192 z\u0305)', 'Decode (z\u0305 \u2192 x\u0302)',
                    'Inference (z \u2192 D(z))'])

    # Styling
    ax.set_xlim(-0.15, 1.15)
    all_values = np.concatenate([z_prior, x_data, z_, x_hat, z_infer, x_infer])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Space')
    ax.set_ylabel('Value')
    ax.set_title(f'NO-VAE: Noise Oriented VAE (T={temperature})')
    ax.set_xticks([PLOT_PARAMS['left_x'], PLOT_PARAMS['right_x']])
    ax.set_xticklabels(['Latent (z, z\u0305)', 'Data (x, x\u0302)'])
    ax.legend(handles=handles, labels=labels, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"NO-VAE detailed visualization saved to {save_path}")

    return fig, ax


def visualize_novae_1d(z_infer, x_infer, x_data, x_recon=None, save_path=None, epoch=None):
    """
    NO-VAE 1D visualization in comparison format (similar to flow model viz).

    Shows straight lines from z (prior) to D(z) (generated), illustrating
    the one-step nature of NO-VAE vs multi-step ODE of flow models.

    Args:
        z_infer: Prior samples z (n_infer,)
        x_infer: Generated samples decoder(z) (n_infer,)
        x_data: Training data x (n_train,)
        x_recon: Reconstruction of training data x_data (n_train,) - optional
        save_path: Path to save figure
        epoch: Current epoch (for title)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=PLOT_PARAMS['figsize'])

    n_infer = len(z_infer)

    # Plot straight lines from z (t=0) to x_hat (t=1) - one-step mapping
    for i in range(n_infer):
        ax.plot(
            [0, 1],
            [z_infer[i], x_infer[i]],
            color=COLORS['trajectory'],
            alpha=0.5,
            linewidth=PLOT_PARAMS['line_width'] * 1.5,
            zorder=2,
            label='z \u2192 D(z)' if i == 0 else None
        )

    # Plot z (start) points at t=0
    ax.scatter(
        np.zeros(n_infer), z_infer,
        color=COLORS['source_x0_infer'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='z (prior)',
        zorder=3
    )

    # Plot x_hat (generated) points at t=1
    ax.scatter(
        np.ones(n_infer), x_infer,
        color=COLORS['infer_x_hat'], marker='o',
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x\u0302 (generated)',
        zorder=3
    )

    # Plot training data at t=1 for reference
    ax.scatter(
        np.ones_like(x_data), x_data,
        color=COLORS['data_x'], marker=PLOT_PARAMS['marker_data_x'],
        s=PLOT_PARAMS['point_size'],
        alpha=PLOT_PARAMS['point_alpha'],
        label='x (train data)',
        zorder=3
    )
    
    # Plot reconstruction of training data if provided
    if x_recon is not None:
        ax.scatter(
            np.ones_like(x_recon), x_recon,
            color=COLORS['output_x_hat'], marker='o',
            s=PLOT_PARAMS['point_size'],
            alpha=PLOT_PARAMS['point_alpha'],
            label='x\u0302 (recon)',
            zorder=3
        )

    # Styling
    all_values = np.concatenate([z_infer, x_infer, x_data])
    if x_recon is not None:
        all_values = np.concatenate([all_values, x_recon])
    y_min, y_max = all_values.min(), all_values.max()
    ax.set_ylim(y_min - PLOT_PARAMS['y_padding'], y_max + PLOT_PARAMS['y_padding'])

    ax.set_xlabel('Space')
    ax.set_ylabel('Value')
    title = 'NO-VAE: z \u2192 D(z) One-Step'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Latent (z)', 'Data (x\u0302)'])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None and epoch % 50 == 0:
            print(f"NO-VAE epoch {epoch} visualization saved to {save_path}")

    return fig, ax


if __name__ == "__main__":
    # Test visualizations with dummy data
    print("Testing NO-VAE 1D visualizations...")

    n_samples = 100
    n_infer = 50

    # Generate dummy data
    z_prior = np.random.randn(n_samples)
    x_data = np.random.randn(n_samples) * 0.5 + 1.5
    z_ = x_data * 0.4 + np.random.randn(n_samples) * 0.3
    x_hat = x_data + np.random.randn(n_samples) * 0.2
    z_infer = np.random.randn(n_infer)
    x_infer = z_infer * 0.3 + 1.5 + np.random.randn(n_infer) * 0.3

    # Test detailed visualization
    fig, ax = visualize_novae(
        z_prior, x_data, z_, x_hat, z_infer, x_infer,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_novae_detailed.png'
    )
    plt.close()

    # Test comparison format
    fig, ax = visualize_novae_1d(
        z_infer, x_infer, x_data,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_novae_1d.png',
        epoch=100
    )
    plt.close()

    print("Test complete!")
