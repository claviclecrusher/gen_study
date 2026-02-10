"""
Visualization for Noise Oriented VAE (NO-VAE) - 2D

Two separate visualizations:
1. Reconstruction: x → encoder → z' → soft NN → z → decoder → x'
2. Generation: z (prior) → decoder → x' (generated)
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style

# Filter out matplotlib warnings for unfilled markers with edgecolors
warnings.filterwarnings('ignore', category=UserWarning, message='.*edgecolor.*unfilled marker.*')


def visualize_novae_2d_recon(x_data, z_, z_selected, x_recon, save_path=None, epoch=None):
    """
    Visualize NO-VAE reconstruction pipeline in 2D: x → z' → z → x'
    
    Shows the full reconstruction process:
    - x (training data)
    - z' (encoder output)
    - z (selected via soft nearest neighbor)
    - x' (reconstruction)
    - Lines: x → z', z' → z, z → x'
    
    Args:
        x_data: Training data x (n_train, 2)
        z_: Encoder output z' (n_train, 2)
        z_selected: Selected z via soft NN (n_train, 2)
        x_recon: Reconstruction x' (n_train, 2)
        save_path: Path to save figure
        epoch: Current epoch (for title)
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    n_samples = len(x_data)
    
    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
               color='black', alpha=0.3, s=15, marker='x',
               label='x (data)', zorder=1)
    
    # Plot lines: x → z' (encoding)
    for i in range(n_samples):
        ax.plot([x_data[i, 0], z_[i, 0]],
                [x_data[i, 1], z_[i, 1]],
                color=COLORS['encode_line'], alpha=0.3, linewidth=0.8,
                linestyle='--', zorder=2)
    
    # Plot lines: z' → z (soft nearest neighbor selection)
    for i in range(n_samples):
        ax.plot([z_[i, 0], z_selected[i, 0]],
                [z_[i, 1], z_selected[i, 1]],
                color='orange', alpha=0.4, linewidth=0.8,
                linestyle=':', zorder=2)
    
    # Plot lines: z → x' (decoding)
    for i in range(n_samples):
        ax.plot([z_selected[i, 0], x_recon[i, 0]],
                [z_selected[i, 1], x_recon[i, 1]],
                color=COLORS['decode_line'], alpha=0.3, linewidth=0.8,
                linestyle='--', zorder=2)
    
    # Plot points
    ax.scatter(x_data[:, 0], x_data[:, 1],
               color=COLORS['data_x'], alpha=0.7, s=30, marker='x',
               label='x (data)', zorder=3)
    
    ax.scatter(z_[:, 0], z_[:, 1],
               color=COLORS['latent_z_hat'], alpha=0.7, s=25, marker='o',
               edgecolors='white', linewidths=0.5, label="z' (encoder)", zorder=3)
    
    ax.scatter(z_selected[:, 0], z_selected[:, 1],
               color='orange', alpha=0.7, s=25, marker='s',
               edgecolors='white', linewidths=0.5, label='z (selected)', zorder=3)
    
    ax.scatter(x_recon[:, 0], x_recon[:, 1],
               color=COLORS['output_x_hat'], alpha=0.7, s=30, marker='o',
               edgecolors='white', linewidths=0.5, label="x' (recon)", zorder=3)
    
    ax.set_xlabel('x\u2081')
    ax.set_ylabel('x\u2082')
    title = 'NO-VAE: Reconstruction Pipeline'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set limits
    all_x = np.concatenate([x_data[:, 0], z_[:, 0], z_selected[:, 0], x_recon[:, 0]])
    all_y = np.concatenate([x_data[:, 1], z_[:, 1], z_selected[:, 1], x_recon[:, 1]])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None and epoch % 50 == 0:
            print(f"NO-VAE 2D reconstruction epoch {epoch} visualization saved to {save_path}")
    
    return fig, ax


def visualize_novae_2d_gen(z_infer, x_infer, x_data, save_path=None, epoch=None):
    """
    Visualize NO-VAE generation in 2D: z (prior) → decoder → x' (generated)
    
    Args:
        z_infer: Prior samples z (n_infer, 2)
        x_infer: Generated samples decoder(z) (n_infer, 2)
        x_data: Training data x (n_train, 2) for reference
        save_path: Path to save figure
        epoch: Current epoch (for title)
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    n_infer = len(z_infer)
    
    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
               color='black', alpha=0.3, s=15, marker='x',
               label='Real data', zorder=1)
    
    # Plot lines from z to D(z) (one-step mapping)
    for i in range(n_infer):
        ax.plot([z_infer[i, 0], x_infer[i, 0]],
                [z_infer[i, 1], x_infer[i, 1]],
                color=COLORS['trajectory'], alpha=0.5, linewidth=1.0, zorder=2)
    
    # Plot z (start) points
    ax.scatter(z_infer[:, 0], z_infer[:, 1],
               color=COLORS['source_x0'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='z (prior)', zorder=3)
    
    # Plot x_hat (generated) points
    ax.scatter(x_infer[:, 0], x_infer[:, 1],
               color=COLORS['infer_x_hat'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='x\u0302 (generated)', zorder=3)
    
    ax.set_xlabel('x\u2081')
    ax.set_ylabel('x\u2082')
    title = 'NO-VAE: Generation'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set limits based on all data with padding
    all_x = np.concatenate([x_data[:, 0], z_infer[:, 0], x_infer[:, 0]])
    all_y = np.concatenate([x_data[:, 1], z_infer[:, 1], x_infer[:, 1]])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None and epoch % 50 == 0:
            print(f"NO-VAE 2D generation epoch {epoch} visualization saved to {save_path}")
    
    return fig, ax


def visualize_novae_2d(z_infer, x_infer, x_data, x_recon=None, save_path=None, epoch=None):
    """
    Visualize NO-VAE in 2D with one-step mapping lines.

    Args:
        z_infer: Prior samples z (n_infer, 2)
        x_infer: Generated samples decoder(z) (n_infer, 2)
        x_data: Training data x (n_train, 2)
        x_recon: Reconstruction of training data x_data (n_train, 2) - optional
        save_path: Path to save figure
        epoch: Current epoch (for title)
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    n_infer = len(z_infer)

    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
               color='black', alpha=0.3, s=15, marker='x',
               label='Real data', zorder=1)
    
    # Plot reconstruction of training data if provided
    if x_recon is not None:
        ax.scatter(x_recon[:, 0], x_recon[:, 1],
                   color=COLORS['output_x_hat'], alpha=0.6, s=25, marker='o',
                   edgecolors='white', linewidths=0.5,
                   label='x\u0302 (recon)', zorder=2)

    # Plot lines from z to D(z) (one-step mapping)
    for i in range(n_infer):
        ax.plot([z_infer[i, 0], x_infer[i, 0]],
                [z_infer[i, 1], x_infer[i, 1]],
                color=COLORS['trajectory'], alpha=0.5, linewidth=1.0, zorder=2)

    # Plot z (start) points
    ax.scatter(z_infer[:, 0], z_infer[:, 1],
               color=COLORS['source_x0'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='z (prior)', zorder=3)

    # Plot x_hat (generated) points
    ax.scatter(x_infer[:, 0], x_infer[:, 1],
               color=COLORS['infer_x_hat'], alpha=0.8, s=30, edgecolors='white',
               linewidths=0.5, label='x\u0302 (generated)', zorder=3)

    ax.set_xlabel('x\u2081')
    ax.set_ylabel('x\u2082')
    title = 'NO-VAE (One-Step)'
    if epoch is not None:
        title += f' (Epoch {epoch})'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set limits based on all data with padding
    all_x = np.concatenate([x_data[:, 0], z_infer[:, 0], x_infer[:, 0]])
    all_y = np.concatenate([x_data[:, 1], z_infer[:, 1], x_infer[:, 1]])
    if x_recon is not None:
        all_x = np.concatenate([all_x, x_recon[:, 0]])
        all_y = np.concatenate([all_y, x_recon[:, 1]])
    x_padding = (all_x.max() - all_x.min()) * 0.1
    y_padding = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_padding, all_x.max() + x_padding)
    ax.set_ylim(all_y.min() - y_padding, all_y.max() + y_padding)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        if epoch is not None and epoch % 50 == 0:
            print(f"NO-VAE 2D epoch {epoch} visualization saved to {save_path}")

    return fig, ax


if __name__ == '__main__':
    print("Testing NO-VAE 2D visualization...")
    from data.synthetic import generate_data_2d, sample_prior

    n_train, n_infer = 500, 50
    x_data = generate_data_2d(n_samples=n_train, seed=42)

    # Create fake data for testing
    z_infer = sample_prior(n_samples=n_infer, seed=43, dim=2)
    x_infer = z_infer + np.array([1.5, 1.5]) + np.random.randn(n_infer, 2) * 0.3

    visualize_novae_2d(
        z_infer=z_infer,
        x_infer=x_infer,
        x_data=x_data,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_novae_2d.png',
        epoch=1
    )
    print("Test complete!")
