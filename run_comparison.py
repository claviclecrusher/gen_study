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
        'modefm': 'ModeFM',
        'meanflow': 'MeanFlow',
        'imf': 'iMF (Improved MeanFlow)',
        'tdmf': 'TDMF (Translation Decoupled)',
        'facm': 'FACM',
        'backflow': 'BackFlow',
        'topk_fm': 'TopK-OTCFM',
        'novae': 'NO-VAE'
    }

    # Models that use CFM coupling (NO-VAE does not)
    cfm_models = {'fm', 'modefm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm'}

    # Filter models that have images for this epoch
    models_with_images = []
    for model_key in available_models:
        # Extract model name and CFM type from model_key
        parts = model_key.split('_')
        cfm_types_list = ['icfm', 'otcfm', 'uotcfm', 'uotrfm']
        
        # Find CFM type
        cfm_type = None
        cfm_idx = -1
        for i, part in enumerate(parts):
            if part in cfm_types_list:
                cfm_type = part
                cfm_idx = i
                break
        
        # Extract model name
        if cfm_idx > 0:
            model_name = '_'.join(parts[:cfm_idx])
        else:
            model_name = model_key
            for m in sorted(model_title_map.keys(), key=len, reverse=True):
                if model_key.startswith(m):
                    model_name = m
                    break
        
        # Check if image exists
        if model_name == 'topk_fm':
            img_path = os.path.join(model_dirs[model_key], f'topk_fm_epoch_{epoch:04d}.png')
        elif model_name in ('fm', 'modefm'):
            img_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}.png')
        elif model_name == 'novae':
            img_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}_recon.png')
        else:
            img_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}.png')
        
        if os.path.exists(img_path):
            models_with_images.append((model_key, model_name, cfm_type, img_path))
    
    # Skip if no models have images for this epoch
    if len(models_with_images) == 0:
        return
    
    # Recreate figure with only models that have images
    n_models_with_images = len(models_with_images)
    fig, axes = plt.subplots(1, n_models_with_images, figsize=(5*n_models_with_images, 4.5))
    if n_models_with_images == 1:
        axes = [axes]  # Make it iterable
    
    plt.subplots_adjust(wspace=subplot_spacing)
    
    for ax, (model_key, model_name, cfm_type, img_path) in zip(axes, models_with_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_grid_recon_gen(model_dirs, epoch, output_path, dim='1d', subplot_spacing=0.0):
    """
    Create a grid comparing auto-encoder based models with recon and gen side-by-side
    Args:
        model_dirs: Dict with model names as keys and directories as values
        epoch: Epoch number
        output_path: Path to save the comparison image
        dim: '1d' or '2d'
        subplot_spacing: Horizontal spacing between subplots (default: 0.0)
    """
    # Auto-encoder based methods (have separate recon and gen visualizations)
    ae_models = {'novae'}  # Add other auto-encoder models here if needed

    # Filter to only auto-encoder models
    available_models = [k for k in model_dirs.keys() 
                       if any(k.startswith(m) or k.split('_')[0] == m for m in ae_models)]

    if len(available_models) == 0:
        print("No auto-encoder models to compare")
        return

    model_title_map = {
        'novae': 'NO-VAE'
    }

    # Models that use CFM coupling (NO-VAE does not)
    cfm_models = {'fm', 'modefm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm'}

    # Filter models that have at least one image (recon or gen) for this epoch
    models_with_images = []
    for model_key in available_models:
        # Extract model name and CFM type from model_key
        parts = model_key.split('_')
        cfm_types_list = ['icfm', 'otcfm', 'uotcfm', 'uotrfm']

        # Find CFM type
        cfm_type = None
        cfm_idx = -1
        for i, part in enumerate(parts):
            if part in cfm_types_list:
                cfm_type = part
                cfm_idx = i
                break

        # Extract model name
        if cfm_idx > 0:
            model_name = '_'.join(parts[:cfm_idx])
        else:
            model_name = model_key
            for m in sorted(model_title_map.keys(), key=len, reverse=True):
                if model_key.startswith(m):
                    model_name = m
                    break

        # Check if at least one image exists
        recon_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}_recon.png')
        gen_path = os.path.join(model_dirs[model_key], f'epoch_{epoch:04d}_gen.png')

        if os.path.exists(recon_path) or os.path.exists(gen_path):
            models_with_images.append((model_key, model_name, cfm_type, recon_path, gen_path))

    # Skip if no models have images for this epoch
    if len(models_with_images) == 0:
        return

    # Create subplots: 2 columns (recon, gen) for each model
    n_models_with_images = len(models_with_images)
    fig, axes = plt.subplots(n_models_with_images, 2, figsize=(10, 4.5*n_models_with_images))
    if n_models_with_images == 1:
        axes = axes.reshape(1, -1)  # Make it 2D

    plt.subplots_adjust(wspace=subplot_spacing, hspace=0.3)

    for row_idx, (model_key, model_name, cfm_type, recon_path, gen_path) in enumerate(models_with_images):
        # Left subplot: Reconstruction
        ax_recon = axes[row_idx, 0]
        if os.path.exists(recon_path):
            img_recon = Image.open(recon_path)
            ax_recon.imshow(img_recon)
        ax_recon.axis('off')

        # Right subplot: Generation
        ax_gen = axes[row_idx, 1]
        if os.path.exists(gen_path):
            img_gen = Image.open(gen_path)
            ax_gen.imshow(img_gen)
        ax_gen.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiments(dim='1d', epochs=100, n_samples=500, lr=1e-3, batch_size=64,
                   seed=42, device='cpu', models=['fm', 'meanflow', 'imf', 'facm', 'backflow'],
                   cfm_types=None, cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
                   lambda_trans=0.1, lambda_schedule='fixed',
                   topk_pretrain_epochs=150, top_filter_k=0.5,
                   top_filter_k_schedule='fixed', top_filter_k_start=1.0, top_filter_k_end=0.1,
                   ode_solver='dopri5', ode_tol=1e-5, dataset_2d='2gauss',
                   lr_scheduler='cosine', lr_scheduler_params=None,
                   modefm_initial_sigma=None, modefm_min_sigma=0.1, modefm_sigma_decay_factor=0.95,
                   modefm_sigma_schedule='cosine', modefm_sigma_schedule_params=None,
                   modefm_var_head=False, modefm_var_loss_weight=1.0, modefm_sigma_scale=1.0,
                   sample_adaptive_warmup_epochs=0, sample_adaptive_warmup_sigma=10.0,
                   novae_use_soft_bridging=False, novae_z_recon_weight=1.0,
                   novae_bridging_method='sinkhorn', novae_n_prior_samples=None,
                   novae_n_prior_samples_recon=None, novae_no_sampling_ratio=0.0,
                   novae_beta=0.0, novae_nep_weight=0.0, novae_nep_var_weight=0.0,
                   novae_count_var_weight=0.1):
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
    # Validate and set default CFM types (modefm defaults to otcfm, others to icfm)
    if cfm_types is None:
        cfm_types = ['otcfm' if m == 'modefm' else 'icfm' for m in models]
    elif len(cfm_types) != len(models):
        raise ValueError(f"Number of CFM types ({len(cfm_types)}) must match number of models ({len(models)})")
    
    print("\n" + "="*80)
    print(f"Running {dim.upper()} Experiments")
    print(f"Models: {models}")
    print(f"CFM Types: {cfm_types}")
    if 'modefm' in models:
        sigma_info = f"sigma: init={modefm_initial_sigma or 'auto'}, min={modefm_min_sigma}, schedule={modefm_sigma_schedule}"
        if modefm_var_head:
            sigma_info += f", var_head=on, var_loss_weight={modefm_var_loss_weight}, sigma_scale={modefm_sigma_scale}"
        print(f"ModeFM: {sigma_info}")
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
        
        # For novae, add bridging method to folder name
        if model_name == 'novae':
            model_key = f'{model_key}_{novae_bridging_method}'
        
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

        elif model_name == 'modefm':
            from training.train_modefm import train_modefm
            model, _ = train_modefm(
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
                lr_scheduler_params=lr_scheduler_params,
                initial_sigma=modefm_initial_sigma,
                min_sigma=modefm_min_sigma,
                sigma_decay_factor=modefm_sigma_decay_factor,
                sigma_schedule=modefm_sigma_schedule,
                sigma_schedule_params=modefm_sigma_schedule_params,
                use_var_head=modefm_var_head,
                var_loss_weight=modefm_var_loss_weight,
                sigma_scale=modefm_sigma_scale,
                sample_adaptive_warmup_epochs=sample_adaptive_warmup_epochs,
                sample_adaptive_warmup_sigma=sample_adaptive_warmup_sigma
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

        elif model_name == 'novae':
            from training.train_novae import train_novae
            model, _ = train_novae(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                viz_freq=1,  # Every epoch
                save_dir=model_dir,
                dim=dim,
                dataset_2d=dataset_2d if dim == '2d' else '2gauss',
                bridging_method=novae_bridging_method,
                n_prior_samples=novae_n_prior_samples,
                n_prior_samples_recon=novae_n_prior_samples_recon,
                sinkhorn_reg=0.05,
                sinkhorn_reg_schedule='cosine',
                sinkhorn_reg_init=1.0,
                sinkhorn_reg_final=0.01,
                sinkhorn_use_soft_bridging=novae_use_soft_bridging,
                z_recon_weight=novae_z_recon_weight,
                no_sampling_ratio=novae_no_sampling_ratio,
                beta=novae_beta,
                nep_weight=novae_nep_weight,
                nep_var_weight=novae_nep_var_weight,
                count_var_weight=novae_count_var_weight,
                temperature=0.1,
                temperature_schedule='cosine',
                temperature_init=0.01,
                temperature_final=0.0,
                use_ste=False,
                alignment_weight=1.0,
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

    # Extract NOVAE bridging method from model_dirs for GIF naming
    novae_bridging_method_for_gif = None
    for key in model_dirs.keys():
        if 'novae' in key:
            # Extract bridging method from key (format: novae_icfm_{bridging_method})
            bridging_methods = ['ot_guided_soft', 'sinkhorn', 'softnn', 'inv_softnn']
            for method in bridging_methods:
                if method in key:
                    novae_bridging_method_for_gif = method
                    break
            if novae_bridging_method_for_gif:
                break
    
    # Create GIF from comparison frames
    if novae_bridging_method_for_gif:
        gif_path = os.path.join(save_dir, f'comparison_{dim}_{novae_bridging_method_for_gif}.gif')
    else:
        gif_path = os.path.join(save_dir, f'comparison_{dim}.gif')
    create_gif(comparison_dir, gif_path, fps=10)

    # Check if there are auto-encoder based models and create recon_gen GIF
    ae_models = {'novae'}  # Auto-encoder based methods
    has_ae_model = any(any(k.startswith(m) or k.split('_')[0] == m for m in ae_models) 
                       for k in model_dirs.keys())
    
    if has_ae_model:
        print("\nCreating recon+gen comparison for auto-encoder models...")
        recon_gen_dir = os.path.join(save_dir, 'comparison_frames_recon_gen')
        os.makedirs(recon_gen_dir, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            recon_gen_path = os.path.join(recon_gen_dir, f'epoch_{epoch:04d}.png')
            create_comparison_grid_recon_gen(model_dirs, epoch, recon_gen_path, dim=dim)
        
        # Create GIF from recon_gen frames
        if novae_bridging_method_for_gif:
            recon_gen_gif_path = os.path.join(save_dir, f'comparison_{dim}_recon_gen_{novae_bridging_method_for_gif}.gif')
        else:
            recon_gen_gif_path = os.path.join(save_dir, f'comparison_{dim}_recon_gen.gif')
        create_gif(recon_gen_dir, recon_gen_gif_path, fps=10)

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
    # Validate and set default CFM types (modefm defaults to otcfm, others to icfm)
    if cfm_types is None:
        cfm_types = ['otcfm' if m == 'modefm' else 'icfm' for m in models]
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
        
        # For novae, try with bridging method first (new format)
        if model_name == 'novae':
            # Try different bridging methods
            bridging_methods = ['sinkhorn', 'softnn', 'ot_guided_soft', 'inv_softnn']
            found = False
            for bridging_method in bridging_methods:
                novae_model_key = f'{model_key}_{bridging_method}'
                novae_model_dir = os.path.join(save_dir, f'{novae_model_key}_epochs')
                if os.path.exists(novae_model_dir):
                    model_dirs[model_key] = novae_model_dir
                    print(f"Found {model_name} ({cfm_type}, {bridging_method}) directory: {novae_model_dir}")
                    found = True
                    break
            
            if not found:
                # Try old format without bridging method (backward compatibility)
                model_dir = os.path.join(save_dir, f'{model_key}_epochs')
                if os.path.exists(model_dir):
                    model_dirs[model_key] = model_dir
                    print(f"Found {model_name} ({cfm_type}) directory (old format): {model_dir}")
                else:
                    print(f"Warning: {model_name} ({cfm_type}) directory not found")
        else:
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
    
    # Extract NOVAE bridging method from model_dirs for GIF naming
    novae_bridging_method_for_gif = None
    for key in model_dirs.keys():
        if 'novae' in key:
            # Extract bridging method from key (format: novae_icfm_{bridging_method})
            bridging_methods = ['ot_guided_soft', 'sinkhorn', 'softnn', 'inv_softnn']
            for method in bridging_methods:
                if method in key:
                    novae_bridging_method_for_gif = method
                    break
            if novae_bridging_method_for_gif:
                break
    
    if novae_bridging_method_for_gif:
        gif_path = os.path.join(save_dir, f'comparison_{dim}_{novae_bridging_method_for_gif}.gif')
    else:
        gif_path = os.path.join(save_dir, f'comparison_{dim}.gif')
    create_gif(comparison_dir, gif_path, fps=10)
    
    # Check if there are auto-encoder based models and create recon_gen GIF
    ae_models = {'novae'}  # Auto-encoder based methods
    has_ae_model = any(any(k.startswith(m) or k.split('_')[0] == m for m in ae_models) 
                       for k in model_dirs.keys())
    
    if has_ae_model:
        print("\nCreating recon+gen comparison for auto-encoder models...")
        recon_gen_dir = os.path.join(save_dir, 'comparison_frames_recon_gen')
        os.makedirs(recon_gen_dir, exist_ok=True)
        
        for epoch in range(1, epochs + 1):
            recon_gen_path = os.path.join(recon_gen_dir, f'epoch_{epoch:04d}.png')
            create_comparison_grid_recon_gen(model_dirs, epoch, recon_gen_path, dim=dim)
            if (epoch % 50 == 0) or epoch == epochs:
                print(f"Created recon+gen comparison image for epoch {epoch}/{epochs}")
        
        # Create GIF from recon_gen frames
        if novae_bridging_method_for_gif:
            recon_gen_gif_path = os.path.join(save_dir, f'comparison_{dim}_recon_gen_{novae_bridging_method_for_gif}.gif')
        else:
            recon_gen_gif_path = os.path.join(save_dir, f'comparison_{dim}_recon_gen.gif')
        create_gif(recon_gen_dir, recon_gen_gif_path, fps=10)
    
    print(f"\n{dim.upper()} visualizations complete!")
    print(f"Results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run 1D/2D comparison experiments')
    parser.add_argument('--dim', type=str, default='2d', choices=['1d', '2d', 'both'],
                       help='Dimension to run (1d, 2d, or both)')
    parser.add_argument('--epochs', type=int, default=400,
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
                       choices=['fm', 'modefm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm', 'novae'],
                       help='Models to train')
    parser.add_argument('--cfm', type=str, nargs='+', default=None,
                       choices=['icfm', 'otcfm', 'uotcfm', 'uotrfm'],
                       help='CFM coupling type(s) - one per model. If not specified: otcfm for modefm, icfm for others. '
                            'If one value given, applied to all models.')
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
    parser.add_argument('--modefm_initial_sigma', type=float, default=None,
                       help='ModeFM: initial sigma for Gaussian kernel (default: 5.0 for 1D, 10.0 for 2D)')
    parser.add_argument('--modefm_min_sigma', type=float, default=0.1,
                       help='ModeFM: minimum sigma for annealing (default: 0.1)')
    parser.add_argument('--modefm_sigma_schedule', type=str, default='cosine',
                       choices=['exponential', 'cosine', 'linear', 'step', 'warmup_cosine', 'warmup_linear',
                                'three_phase_linear', 'sigmoid', 'batch_adaptive', 'sample_adaptive'],
                       help='ModeFM: sigma schedule (sample_adaptive=var_head output, requires --var_head)')
    parser.add_argument('--modefm_sigma_decay_factor', type=float, default=0.95,
                       help='ModeFM: decay factor for exponential schedule (default: 0.95)')
    parser.add_argument('--modefm_sigma_warmup_ratio', type=float, default=0.2,
                       help='ModeFM: warmup ratio for warmup_* schedules (default: 0.2)')
    parser.add_argument('--modefm_sigma_hold_ratio', type=float, default=1.0/3,
                       help='ModeFM: hold ratio for three_phase_linear/sigmoid, phase1 at init (default: 1/3)')
    parser.add_argument('--modefm_sigma_decay_ratio', type=float, default=1.0/3,
                       help='ModeFM: decay ratio for three_phase_linear/sigmoid, phase2 transition (default: 1/3)')
    parser.add_argument('--modefm_sigma_steepness', type=float, default=8.0,
                       help='ModeFM: sigmoid steepness k for sigmoid schedule (default: 8.0)')
    parser.add_argument('--modefm_sigma_gamma', type=float, default=1.0,
                       help='ModeFM: bandwidth scaling for batch_adaptive (default: 1.0)')
    parser.add_argument('--modefm_sigma_q', type=float, default=0.5,
                       help='ModeFM: quantile (0.5=median) for batch_adaptive (default: 0.5)')
    parser.add_argument('--modefm_sigma_eps', type=float, default=1e-6,
                       help='ModeFM: min sigma epsilon for batch_adaptive (default: 1e-6)')
    parser.add_argument('--var_head', type=str, default='false', choices=['true', 'false'],
                       help='ModeFM: add variance head for sample-adaptive sigma (default: false)')
    parser.add_argument('--modefm_var_loss_weight', type=float, default=1.0,
                       help='ModeFM: weight for variance head loss (default: 1.0)')
    parser.add_argument('--modefm_sigma_scale', type=float, default=1.0,
                       help='ModeFM: kernel_width scale for sample_adaptive (default: 1.0)')
    parser.add_argument('--sample_adaptive_warmup_epochs', type=int, default=0,
                       help='ModeFM: epochs to use fixed sigma before sample_adaptive (default: 0)')
    parser.add_argument('--sample_adaptive_warmup_sigma', type=float, default=10.0,
                       help='ModeFM: fixed sigma during sample_adaptive warmup (default: 10.0)')
    parser.add_argument('--dataset_2d', type=str, default='shifted_2gauss',
                       choices=['2gauss', 'shifted_2gauss', 'two_moon'],
                       help='2D dataset type (default: 2gauss)')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['none', 'cosine', 'step', 'exponential'],
                       help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--novae_use_soft_bridging', type=str, default='false',
                       choices=['true', 'false', 'auto'],
                       help='NO-VAE bridging mode: true (soft), false (hard, default), auto (based on reg)')
    parser.add_argument('--novae_use_soft_coupling', type=str, default=None,
                       help=argparse.SUPPRESS)  # deprecated alias for novae_use_soft_bridging
    parser.add_argument('--novae_z_recon_weight', type=float, default=1.0,
                       help='NO-VAE z reconstruction loss weight (default: 1.0)')
    parser.add_argument('--novae_bridging_method', type=str, default='ot_guided_soft',
                       choices=['sinkhorn', 'softnn', 'ot_guided_soft', 'nep', 'inv_softnn'],
                       help='NO-VAE bridging method: sinkhorn, softnn, ot_guided_soft, nep, or inv_softnn')
    parser.add_argument('--novae_coupling_method', type=str, default=None,
                       help=argparse.SUPPRESS)  # deprecated alias for novae_bridging_method
    def _parse_novae_n_prior_samples(value):
        """Parse novae_n_prior_samples: int, 'batch_size', 'n_data', or None."""
        if value is None:
            return None
        s = str(value).strip()
        if s.lower() in ('none', ''):
            return None
        try:
            return int(s)
        except ValueError:
            return s.lower()

    parser.add_argument('--novae_n_prior_samples', type=_parse_novae_n_prior_samples,
                       default=None,
                       help='NO-VAE prior samples: int, "batch_size", "n_data" (default: None, uses batch_size)')
    parser.add_argument('--novae_n_prior_samples_recon', type=int, default=None,
                       help='NO-VAE prior samples for recon inference (default: None, uses input batch size = n_samples)')
    parser.add_argument('--novae_no_sampling_ratio', type=float, default=0.0,
                       help='NO-VAE: fraction of batch using z_enc directly (no bridging) for reconstruction (default: 0.0)')
    parser.add_argument('--novae_beta', type=float, default=0.0,
                       help='NO-VAE: weight for regularization loss (KL(z_enc batch || N(0,I)), default: 0.0)')
    parser.add_argument('--novae_nep_weight', type=float, default=0.0,
                       help='NO-VAE: weight for NEP loss (only when bridging_method=nep, default: 0.0)')
    parser.add_argument('--novae_nep_var_weight', type=float, default=0.0,
                       help='NO-VAE: weight for NEP distance variance penalty (default: 0.0)')
    parser.add_argument('--novae_count_var_weight', type=float, default=0.1,
                       help='NO-VAE: Inverted SoftNN load balancing weight (default: 0.1)')
    parser.add_argument('--viz-only', action='store_true',
                       help='Only create visualizations from existing outputs (no training)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Validate CFM types (None = model-specific default: otcfm for modefm, icfm for others)
    if args.cfm is None:
        cfm_types = None
        print("CFM Types: model-specific defaults (otcfm for modefm, icfm for others)")
    elif len(args.cfm) == 1 and len(args.models) > 1:
        # If only one CFM type provided, use it for all models
        cfm_types = args.cfm * len(args.models)
        print(f"CFM Type: {args.cfm[0].upper()} (applied to all models)")
    elif len(args.cfm) != len(args.models):
        raise ValueError(f"Number of CFM types ({len(args.cfm)}) must match number of models ({len(args.models)})")
    else:
        cfm_types = args.cfm
        print(f"CFM Types: {[c.upper() for c in cfm_types]}")
    
    # Parse NO-VAE soft bridging option (with deprecated novae_use_soft_coupling alias)
    use_soft_arg = getattr(args, 'novae_use_soft_coupling', None) or getattr(args, 'novae_use_soft_bridging', 'false')
    if use_soft_arg == 'true':
        novae_use_soft_bridging = True
    elif use_soft_arg == 'false':
        novae_use_soft_bridging = False
    else:  # 'auto'
        novae_use_soft_bridging = None
    if novae_use_soft_bridging is not None:
        mode_str = "soft" if novae_use_soft_bridging else "hard"
        print(f"NO-VAE bridging mode: {mode_str}")

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
                modefm_initial_sigma=args.modefm_initial_sigma,
                modefm_min_sigma=args.modefm_min_sigma,
                modefm_sigma_decay_factor=args.modefm_sigma_decay_factor,
                modefm_sigma_schedule=args.modefm_sigma_schedule,
                modefm_sigma_schedule_params={
                    'warmup_ratio': args.modefm_sigma_warmup_ratio,
                    'hold_ratio': args.modefm_sigma_hold_ratio,
                    'decay_ratio': args.modefm_sigma_decay_ratio,
                    'steepness': args.modefm_sigma_steepness,
                    'gamma': args.modefm_sigma_gamma,
                    'q': args.modefm_sigma_q,
                    'eps': args.modefm_sigma_eps,
                },
                modefm_var_head=(args.var_head == 'true'),
                modefm_var_loss_weight=args.modefm_var_loss_weight,
                modefm_sigma_scale=args.modefm_sigma_scale,
                sample_adaptive_warmup_epochs=args.sample_adaptive_warmup_epochs,
                sample_adaptive_warmup_sigma=args.sample_adaptive_warmup_sigma,
                dataset_2d=args.dataset_2d,
                lr_scheduler=args.lr_scheduler,
                lr_scheduler_params=None,
                novae_use_soft_bridging=novae_use_soft_bridging,
                novae_z_recon_weight=args.novae_z_recon_weight,
                novae_bridging_method=getattr(args, 'novae_coupling_method', None) or getattr(args, 'novae_bridging_method', 'ot_guided_soft'),
                novae_n_prior_samples=args.novae_n_prior_samples,
                novae_n_prior_samples_recon=args.novae_n_prior_samples_recon,
                novae_no_sampling_ratio=args.novae_no_sampling_ratio,
                novae_beta=args.novae_beta,
                novae_nep_weight=args.novae_nep_weight,
                novae_nep_var_weight=args.novae_nep_var_weight,
                novae_count_var_weight=args.novae_count_var_weight
            )

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
