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
        'imf': 'iMF (Improved MeanFlow)',
        'tdmf': 'TDMF (Translation Decoupled)',
        'facm': 'FACM',
        'backflow': 'BackFlow',
        'topk_fm': 'TopK-OTCFM'
    }

    for ax, model_key in zip(axes, available_models):
        # Extract model name and CFM type from model_key
        # Examples: 'fm_icfm' -> ('fm', 'icfm'), 'topk_fm_otcfm' -> ('topk_fm', 'otcfm')
        # 'fm_otcfm_0' -> ('fm', 'otcfm') (ignore index suffix)
        parts = model_key.split('_')
        cfm_types_list = ['icfm', 'otcfm', 'uotcfm', 'uotrfm']
        
        # Find CFM type (should be one of the known types)
        cfm_type = None
        cfm_idx = -1
        for i, part in enumerate(parts):
            if part in cfm_types_list:
                cfm_type = part
                cfm_idx = i
                break
        
        # Extract model name (everything before CFM type)
        if cfm_idx > 0:
            model_name = '_'.join(parts[:cfm_idx])
        else:
            # Fallback: try to match known model names
            model_name = model_key
            for m in sorted(model_title_map.keys(), key=len, reverse=True):
                if model_key.startswith(m):
                    model_name = m
                    break
        
        title = model_title_map.get(model_name, model_name)
        # Add CFM type to title if found
        if cfm_type:
            title = f'{title} ({cfm_type.upper()})'
        
        # TopK-OTCFM uses different filename format
        if model_name == 'topk_fm':
            img_path = os.path.join(model_dirs[model_key], f'topk_fm_epoch_{epoch:04d}.png')
        else:
            img_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}.png')

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
                   seed=42, device='cpu', models=['fm', 'meanflow', 'imf', 'facm', 'backflow'],
                   cfm_types=None, cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
                   lambda_trans=0.1, lambda_schedule='fixed',
                   topk_pretrain_epochs=150, top_filter_k=0.5, 
                   top_filter_k_schedule='fixed', top_filter_k_start=1.0, top_filter_k_end=0.1,
                   ode_solver='dopri5', ode_tol=1e-5, dataset_2d='2gauss',
                   lr_scheduler='cosine', lr_scheduler_params=None):
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
        cfm_types: List of CFM coupling types ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
                   Should match the length of models list. If None, defaults to 'icfm' for all.
        cfm_reg: Entropic regularization for Sinkhorn
        cfm_reg_m: Marginal regularization for unbalanced OT
        cfm_weight_power: Power factor for UOTRFM weights
    """
    # Validate and set default CFM types
    if cfm_types is None:
        cfm_types = ['icfm'] * len(models)
    elif len(cfm_types) != len(models):
        raise ValueError(f"Number of CFM types ({len(cfm_types)}) must match number of models ({len(models)})")
    
    print("\n" + "="*80)
    print(f"Running {dim.upper()} Experiments")
    print(f"Models: {models}")
    print(f"CFM Types: {cfm_types}")
    print("="*80)

    input_dim = 1 if dim == '1d' else 2
    if dim == '2d':
        save_dir = f'/home/user/Desktop/Gen_Study/outputs/comparison_{dim}_{dataset_2d}'
    else:
        save_dir = f'/home/user/Desktop/Gen_Study/outputs/comparison_{dim}'
    os.makedirs(save_dir, exist_ok=True)

    # Train each model
    model_dirs = {}

    for idx, model_name in enumerate(models):
        cfm_type = cfm_types[idx]
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()} ({dim.upper()}) with {cfm_type.upper()}")
        print(f"{'='*80}")

        # Create unique directory name for each model-cfm combination
        # If same model appears multiple times, add index to make it unique
        model_key = f'{model_name}_{cfm_type}'
        if model_key in model_dirs:
            # If duplicate, add index
            model_key = f'{model_name}_{cfm_type}_{idx}'
        model_dir = os.path.join(save_dir, f'{model_key}_epochs')
        model_dirs[model_key] = model_dir
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
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
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
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
            )

        elif model_name == 'imf':
            from training.train_imf import train_imf
            model, _ = train_imf(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_interval=1,  # Every epoch
                viz_output_dir=model_dir,
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
            )

        elif model_name == 'tdmf':
            from training.train_tdmf import train_tdmf
            model, _ = train_tdmf(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_interval=1,  # Every epoch
                viz_output_dir=model_dir,
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                lambda_trans=lambda_trans,
                lambda_schedule=lambda_schedule,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
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
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
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
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
            )

        elif model_name == 'topk_fm':
            from training.train_topk_fm import train_topk_fm
            model, _ = train_topk_fm(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_freq=1,  # Every epoch
                save_dir=model_dir,
                dim=dim,
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power,
                topk_pretrain_epochs=topk_pretrain_epochs,
                top_filter_k=top_filter_k,
                top_filter_k_schedule=top_filter_k_schedule,
                top_filter_k_start=top_filter_k_start,
                top_filter_k_end=top_filter_k_end,
                ode_solver=ode_solver,
                ode_tol=ode_tol,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                lr_scheduler=lr_scheduler,
                lr_scheduler_params=lr_scheduler_params
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


def create_visualizations_only(dim='1d', epochs=100, models=['fm', 'meanflow', 'imf', 'facm', 'backflow'], cfm_types=None, dataset_2d='2gauss'):
    """
    Create comparison visualizations and GIFs from existing training outputs without training
    
    Args:
        dim: '1d' or '2d'
        epochs: Number of epochs to visualize
        models: List of model names to include in comparison
        cfm_types: List of CFM types used in training (for folder name matching)
                   Should match the length of models list. If None, defaults to 'icfm' for all.
        dataset_2d: 2D dataset type (for folder name matching when dim='2d')
    """
    # Validate and set default CFM types
    if cfm_types is None:
        cfm_types = ['icfm'] * len(models)
    elif len(cfm_types) != len(models):
        raise ValueError(f"Number of CFM types ({len(cfm_types)}) must match number of models ({len(models)})")
    
    print("\n" + "="*80)
    print(f"Creating {dim.upper()} Comparison Visualizations (No Training)")
    print(f"Models: {models}")
    print(f"CFM Types: {cfm_types}")
    if dim == '2d':
        print(f"Dataset: {dataset_2d}")
    print("="*80)
    
    if dim == '2d':
        save_dir = f'/home/user/Desktop/Gen_Study/outputs/comparison_{dim}_{dataset_2d}'
    else:
        save_dir = f'/home/user/Desktop/Gen_Study/outputs/comparison_{dim}'
    
    # Build model directories dictionary
    model_dirs = {}
    for idx, model_name in enumerate(models):
        cfm_type = cfm_types[idx]
        # Create model key (same format as in run_experiments)
        model_key = f'{model_name}_{cfm_type}'
        if model_key in model_dirs:
            model_key = f'{model_name}_{cfm_type}_{idx}'
        
        model_dir = os.path.join(save_dir, f'{model_key}_epochs')
        if os.path.exists(model_dir):
            model_dirs[model_key] = model_dir
            print(f"Found {model_name} ({cfm_type}) directory: {model_dir}")
        else:
            # Try old format for backward compatibility
            old_model_dir = os.path.join(save_dir, f'{model_name}_{cfm_type}_epochs')
            if os.path.exists(old_model_dir):
                model_dirs[model_key] = old_model_dir
                print(f"Found {model_name} ({cfm_type}) directory: {old_model_dir}")
            else:
                old_format_dir = os.path.join(save_dir, f'{model_name}_epochs')
                if os.path.exists(old_format_dir):
                    model_dirs[model_key] = old_format_dir
                    print(f"Found {model_name} directory (old format): {old_format_dir}")
                else:
                    print(f"Warning: {model_name} ({cfm_type}) directory not found: {model_dir}")
    
    if not model_dirs:
        print(f"Error: No model directories found in {save_dir}")
        return
    
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
        if (epoch % 50 == 0) or epoch == epochs:
            print(f"Created comparison image for epoch {epoch}/{epochs}")
    
    # Create GIF from comparison frames
    gif_path = os.path.join(save_dir, f'comparison_{dim}.gif')
    create_gif(comparison_dir, gif_path, fps=10)
    
    print(f"\n{dim.upper()} visualizations complete!")
    print(f"Results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run 1D/2D comparison experiments')
    parser.add_argument('--dim', type=str, default='both', choices=['1d', '2d', 'both'],
                       help='Dimension to run (1d, 2d, or both)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['fm', 'meanflow', 'imf', 'facm', 'backflow'],
                       choices=['fm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm'],
                       help='Models to train')
    parser.add_argument('--cfm', type=str, nargs='+', default=['icfm'],
                       choices=['icfm', 'otcfm', 'uotcfm', 'uotrfm'],
                       help='CFM coupling type(s) - one per model (default: icfm for all). '
                            'If multiple models, provide one CFM type per model in order.')
    parser.add_argument('--cfm_weight_power', type=float, default=10.0,
                       help='Power factor for UOTRFM weights (default: 10.0)')
    parser.add_argument('--lambda_trans', type=float, default=0.1,
                       help='Weight for translation loss in TDMF (default: 0.1)')
    parser.add_argument('--lambda_schedule', type=str, default='fixed',
                       choices=['fixed', 'linear'],
                       help='Lambda schedule type for TDMF (default: fixed)')
    parser.add_argument('--topk_pretrain_epochs', type=int, default=150,
                       help='Number of epochs for TopK-OTCFM pretraining (default: 150)')
    parser.add_argument('--top_filter_k', type=float, default=0.5,
                       help='Fixed top_filter_k value (if schedule=fixed) or initial value (0 < k <= 1, default: 0.5)')
    parser.add_argument('--top_filter_k_schedule', type=str, default='fixed',
                       choices=['fixed', 'linear', 'exponential', 'cosine'],
                       help='Schedule type for top_filter_k (default: fixed)')
    parser.add_argument('--top_filter_k_start', type=float, default=1.0,
                       help='Starting value for top_filter_k at retraining start (default: 1.0)')
    parser.add_argument('--top_filter_k_end', type=float, default=0.1,
                       help='Ending value for top_filter_k at end of training (default: 0.1)')
    parser.add_argument('--ode_solver', type=str, default='dopri5',
                       choices=['dopri5', 'euler'],
                       help='ODE solver method for TopK-OTCFM (default: dopri5)')
    parser.add_argument('--ode_tol', type=float, default=1e-5,
                       help='Tolerance for adaptive ODE solver (default: 1e-5)')
    parser.add_argument('--dataset_2d', type=str, default='2gauss',
                       choices=['2gauss', 'shifted_2gauss', 'two_moon'],
                       help='2D dataset type (default: 2gauss)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step', 'exponential'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--viz-only', action='store_true',
                       help='Only create visualizations from existing outputs (no training)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Validate CFM types
    if len(args.cfm) == 1 and len(args.models) > 1:
        # If only one CFM type provided, use it for all models
        cfm_types = args.cfm * len(args.models)
        print(f"CFM Type: {args.cfm[0].upper()} (applied to all models)")
    elif len(args.cfm) != len(args.models):
        raise ValueError(f"Number of CFM types ({len(args.cfm)}) must match number of models ({len(args.models)})")
    else:
        cfm_types = args.cfm
        print(f"CFM Types: {[c.upper() for c in cfm_types]}")

    dims_to_run = ['1d', '2d'] if args.dim == 'both' else [args.dim]
    
    if args.viz_only:
        # Only create visualizations from existing outputs
        for dim in dims_to_run:
            create_visualizations_only(
                dim=dim,
                epochs=args.epochs,
                models=args.models,
                cfm_types=cfm_types,
                dataset_2d=args.dataset_2d
            )
    else:
        # Run full training and visualization
        for dim in dims_to_run:
            run_experiments(
                dim=dim,
                epochs=args.epochs,
                n_samples=args.n_samples,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
                models=args.models,
                cfm_types=cfm_types,
                cfm_reg=0.05,
                cfm_reg_m=(float('inf'), 2.0),
                cfm_weight_power=args.cfm_weight_power,
                lambda_trans=args.lambda_trans,
                lambda_schedule=args.lambda_schedule,
                topk_pretrain_epochs=args.topk_pretrain_epochs,
                top_filter_k=args.top_filter_k,
                top_filter_k_schedule=args.top_filter_k_schedule,
                top_filter_k_start=args.top_filter_k_start,
                top_filter_k_end=args.top_filter_k_end,
                ode_solver=args.ode_solver,
                ode_tol=args.ode_tol,
                dataset_2d=args.dataset_2d,
                lr_scheduler=args.lr_scheduler,
                lr_scheduler_params=None
            )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
