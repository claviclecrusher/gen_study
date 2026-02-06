"""
Visualization for TopK-OTCFM model in 2D
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization.config import COLORS, PLOT_PARAMS, setup_plot_style


def visualize_topk_fm_2d(trajectories, x_data, save_path=None, epoch=None, is_pretraining=False, cfm_type=None, nfe=None):
    """
    Visualize TopK-OTCFM model in 2D with trajectories
    
    Args:
        trajectories: ODE trajectories from z to x (n_steps+1, n_infer, 2)
        x_data: Training target samples x (n_train, 2)
        save_path: Path to save the figure (optional)
        epoch: Current epoch number (optional, for title)
        is_pretraining: Whether currently in pretraining stage (optional)
        cfm_type: CFM coupling type (optional, for title)
        nfe: Average NFE (Number of Function Evaluations) per batch (optional)
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    n_steps = trajectories.shape[0]
    n_infer = trajectories.shape[1]
    
    # Plot real data (background reference)
    ax.scatter(x_data[:, 0], x_data[:, 1],
              color='black', alpha=0.3, s=15, marker='x',
              label='Real data', zorder=1)
    
    # Plot trajectories (z -> x)
    for i in range(n_infer):
        ax.plot(trajectories[:, i, 0], trajectories[:, i, 1],
               color=COLORS['trajectory'], alpha=0.5, linewidth=1.0, zorder=2)
    
    # Plot starting points (z)
    ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1],
              color=COLORS['source_x0'], alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='z (start)', zorder=3)
    
    # Plot final points (generated x)
    ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1],
              color=COLORS['infer_x_hat'], alpha=0.8, s=30, edgecolors='white',
              linewidths=0.5, label='x (generated)', zorder=3)
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    stage = "Pretraining" if is_pretraining else "Retraining"
    cfm_str = f' ({cfm_type.upper()})' if cfm_type else ''
    title = f'TopK-OTCFM{cfm_str}: {stage} (Epoch {epoch})' if epoch is not None else f'TopK-OTCFM{cfm_str}: {stage}'
    if nfe is not None and not is_pretraining:
        title += f'\nAvg NFE: {nfe:.1f}'
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Set limits based on data range with padding
    all_x = np.concatenate([x_data[:, 0], trajectories[:, :, 0].flatten()])
    all_y = np.concatenate([x_data[:, 1], trajectories[:, :, 1].flatten()])
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
            print(f"TopK-OTCFM 2D visualization saved to {save_path}")
    
    return fig, ax


if __name__ == '__main__':
    print("Testing TopK-OTCFM 2D visualization...")
    from data.synthetic import generate_data_2d, sample_prior
    
    n_train, n_infer = 500, 20
    n_steps = 50
    x_data = generate_data_2d(n_samples=n_train, seed=42)
    
    # Create fake trajectories for testing
    z_init = sample_prior(n_samples=n_infer, seed=43, dim=2)
    x_final = z_init + np.array([1.5, 1.5])
    
    # Linear interpolation for trajectory
    trajectories = np.zeros((n_steps + 1, n_infer, 2))
    for i in range(n_steps + 1):
        t = i / n_steps
        trajectories[i] = (1 - t) * z_init + t * x_final
    
    visualize_topk_fm_2d(
        trajectories=trajectories,
        x_data=x_data,
        save_path='/home/user/Desktop/Gen_Study/outputs/test_viz_topk_fm_2d.png',
        epoch=1,
        is_pretraining=False
    )
    print("Test complete!")
