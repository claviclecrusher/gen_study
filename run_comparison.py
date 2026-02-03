"""
Unified script to run 1D and 2D experiments for FM, MeanFlow, and BackFlow
with visualization at every epoch and GIF generation
"""
import os
import sys
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic import generate_data, generate_data_2d, sample_prior


def create_gif(image_dir, output_path, fps=10):
    """
    Create GIF from images in a directory

    Args:
        image_dir: Directory containing PNG images
        output_path: Path to save the GIF
        fps: Frames per second
    """
    import glob

    # Get all PNG files sorted by name
    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    if not images:
        print(f"No images found in {image_dir}")
        return

    # Load images
    frames = [Image.open(img) for img in images]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0
    )

    print(f"GIF saved to {output_path} ({len(frames)} frames)")


def create_comparison_grid(model_dirs, epoch, output_path, dim='1d', subplot_spacing=0.0):
    """
    Create a grid comparing all three models at a specific epoch

    Args:
        model_dirs: Dict with model names as keys and directories as values
        epoch: Epoch number
        output_path: Path to save the comparison image
        dim: '1d' or '2d'
        subplot_spacing: Horizontal spacing between subplots (default: 0.15)
    """
    available_models = list(model_dirs.keys())
    n_models = len(available_models)

    if n_models == 0:
        print("No models to compare")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4.5))
    if n_models == 1:
        axes = [axes]  # Make it iterable

    # Adjust spacing between subplots (configurable via subplot_spacing parameter)
    plt.subplots_adjust(wspace=subplot_spacing)

    model_title_map = {
        'fm': 'Flow Matching',
        'meanflow': 'MeanFlow',
        'facm': 'FACM',
        'backflow': 'BackFlow'
    }

    for ax, model_name in zip(axes, available_models):
        title = model_title_map.get(model_name, model_name)
        img_path = os.path.join(model_dirs[model_name], f'epoch_{epoch:04d}.png')

        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(f'{title} (Epoch {epoch})')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'No image for epoch {epoch}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.suptitle(f'{dim.upper()} Comparison - Epoch {epoch}', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiments(dim='1d', epochs=100, n_samples=500, lr=1e-3, batch_size=64,
                   seed=42, device='cpu', models=['fm', 'meanflow', 'facm', 'backflow']):
    """
    Run experiments for specified models in 1D or 2D

    Args:
        dim: '1d' or '2d'
        epochs: Number of training epochs
        n_samples: Number of training samples
        lr: Learning rate
        batch_size: Batch size
        seed: Random seed
        device: Device to use
        models: List of models to train ['fm', 'meanflow', 'backflow']
    """
    print("\n" + "="*80)
    print(f"Running {dim.upper()} Experiments")
    print("="*80)

    input_dim = 1 if dim == '1d' else 2
    save_dir = f'/home/user/Desktop/Gen_Study/outputs/comparison_{dim}'
    os.makedirs(save_dir, exist_ok=True)

    # Train each model
    model_dirs = {}

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} ({dim.upper()})")
        print(f"{'='*80}")

        model_dir = os.path.join(save_dir, f'{model_name}_epochs')
        model_dirs[model_name] = model_dir
        os.makedirs(model_dir, exist_ok=True)

        if model_name == 'fm':
            from training.train_fm import train_fm
            model, _ = train_fm(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_freq=1,  # Every epoch
                save_dir=model_dir,
                dim=dim
            )

        elif model_name == 'meanflow':
            from training.train_meanflow import train_meanflow
            model, _ = train_meanflow(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_interval=1,  # Every epoch
                viz_output_dir=model_dir,
                dim=dim
            )

        elif model_name == 'facm':
            from training.train_facm import train_facm
            model, _ = train_facm(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_interval=1,  # Every epoch
                viz_output_dir=model_dir,
                dim=dim
            )

        elif model_name == 'backflow':
            from training.train_backflow import train_backflow
            model, _ = train_backflow(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_freq=1,  # Every epoch
                save_dir=model_dir,
                dim=dim
            )

    # Create comparison GIFs
    print("\n" + "="*80)
    print("Creating comparison visualizations and GIFs")
    print("="*80)

    # Create epoch-by-epoch comparison images
    comparison_dir = os.path.join(save_dir, 'comparison_frames')
    os.makedirs(comparison_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        comparison_path = os.path.join(comparison_dir, f'epoch_{epoch:04d}.png')
        create_comparison_grid(model_dirs, epoch, comparison_path, dim=dim)

    # Create GIF from comparison frames
    gif_path = os.path.join(save_dir, f'comparison_{dim}.gif')
    create_gif(comparison_dir, gif_path, fps=10)

    print(f"\n{dim.upper()} experiments complete!")
    print(f"Results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run 1D/2D comparison experiments')
    parser.add_argument('--dim', type=str, default='both', choices=['1d', '2d', 'both'],
                       help='Dimension to run (1d, 2d, or both)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['fm', 'meanflow', 'facm', 'backflow'],
                       choices=['fm', 'meanflow', 'facm', 'backflow'],
                       help='Models to train')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dims_to_run = ['1d', '2d'] if args.dim == 'both' else [args.dim]

    for dim in dims_to_run:
        run_experiments(
            dim=dim,
            epochs=args.epochs,
            n_samples=args.n_samples,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            models=args.models
        )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
