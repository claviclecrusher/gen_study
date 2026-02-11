"""
Training script for Noise Oriented VAE (NO-VAE)

NO-VAE replaces VAE's Gaussian reparameterization with soft nearest neighbor
selection from prior samples. Training uses only reconstruction loss (MSE)
with no explicit regularization, yet the encoder output distribution
naturally aligns with the prior.
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.novae import NOVAE
from data.synthetic import generate_data, generate_data_2d, sample_prior


def _resolve_n_prior_samples(n_prior_samples, batch_size, n_samples):
    """
    Resolve n_prior_samples to int.
    - int: use as is
    - None: use batch_size (default)
    - 'batch_size': use batch_size
    - 'n_data': use n_samples
    - other str: use batch_size (fallback)
    """
    if n_prior_samples is None:
        return batch_size
    if isinstance(n_prior_samples, int):
        return n_prior_samples
    if isinstance(n_prior_samples, str):
        s = n_prior_samples.strip().lower()
        if s == 'batch_size':
            return batch_size
        if s == 'n_data':
            return n_samples
        return batch_size  # fallback for other strings
    return batch_size  # fallback for other types


def train_novae(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
               n_prior_samples=None, n_prior_samples_recon=None, bridging_method='sinkhorn', sinkhorn_reg=0.05,
               sinkhorn_reg_schedule='cosine', sinkhorn_reg_init=1.0, sinkhorn_reg_final=0.01,
               sinkhorn_use_soft_bridging=False, z_recon_weight=1.0,
               no_sampling_ratio=0.0, beta=0.0, nep_weight=0.0, nep_var_weight=0.0,
               temperature=0.1, temperature_schedule='cosine', temperature_init=0.01, temperature_final=0.0,
               use_ste=False, alignment_weight=1.0, count_var_weight=0.1,
               viz_freq=200, save_dir='/home/user/Desktop/Gen_Study/outputs',
               dim='1d', dataset_2d='2gauss',
               lr_scheduler='cosine', lr_scheduler_params=None):
    """
    Train NO-VAE model

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on
        n_prior_samples: Number of prior samples N per mini-batch. Can be int, str, or None.
            - int: use that value
            - 'batch_size': same as batch_size
            - 'n_data': same as n_samples
            - other str: treated as batch_size
            - None: uses batch_size (default)
        n_prior_samples_recon: Number of prior samples for reconstruction inference. Independent from n_prior_samples.
            - None: use len(viz_x_data) = n_samples, i.e. input batch size at inference (default)
            - int: use that value
        bridging_method: Bridging method ('sinkhorn', default)
        sinkhorn_reg: Entropic regularization for Sinkhorn (if schedule='fixed', default: 0.05)
        sinkhorn_reg_schedule: Epsilon annealing schedule ('fixed', 'linear', 'exponential', 'cosine', default: 'cosine')
        sinkhorn_reg_init: Initial epsilon for annealing (default: 1.0)
        sinkhorn_reg_final: Final epsilon for annealing (default: 0.01)
        sinkhorn_use_soft_bridging: If True, always use soft bridging. If False, always use hard bridging.
                                    If None, auto-select based on sinkhorn_reg (default: None)
        z_recon_weight: Weight for latent reconstruction loss (default: 1.0)
        no_sampling_ratio: Float in [0, 1]. For this fraction of each batch, use z_enc
            directly in the decoder (no bridging with noise) for reconstruction.
            Default: 0.0 (all samples use bridging).
        beta: Weight for regularization loss (default: 0.0). Sample-based KL(z_enc batch || N(0,I)).
        nep_weight: Weight for NEP loss (default: 0.0). Only used when bridging_method='nep'.
        nep_var_weight: Weight for NEP distance variance penalty (default: 0.0).
            Penalizes std of per-sample squared distances between z_enc and z_selected.
        count_var_weight: Weight for Inverted SoftNN load balancing loss (default: 0.1).
            Penalizes std of soft counts per encoder. Only used when bridging_method='inv_softnn'.
        viz_freq: Frequency to save visualizations (every N epochs)
        save_dir: Directory to save visualizations
        dim: Dimension ('1d' or '2d')
        dataset_2d: 2D dataset type ('2gauss', 'shifted_2gauss', 'two_moon')
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Additional scheduler params

    Returns:
        model: Trained NO-VAE model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Resolve n_prior_samples (int, 'batch_size', 'n_data', or None -> batch_size)
    n_prior_resolved = _resolve_n_prior_samples(n_prior_samples, batch_size, n_samples)
    n_prior_display = f"{n_prior_samples} -> {n_prior_resolved}" if isinstance(n_prior_samples, str) else n_prior_resolved

    print("=" * 60)
    print(f"Training Noise Oriented VAE (NO-VAE)")
    n_prior_recon_display = n_prior_samples_recon if n_prior_samples_recon is not None else f"n_samples ({n_samples})"
    print(f"N prior samples: {n_prior_display}")
    print(f"N prior samples (recon inference): {n_prior_recon_display}")
    print(f"Bridging method: {bridging_method}")
    if bridging_method == 'sinkhorn':
        if sinkhorn_reg_schedule == 'fixed':
            print(f"Sinkhorn regularization: {sinkhorn_reg} (fixed)")
        else:
            print(f"Sinkhorn regularization schedule: {sinkhorn_reg_schedule}")
            print(f"  Initial: {sinkhorn_reg_init}, Final: {sinkhorn_reg_final}")
        if sinkhorn_use_soft_bridging is not None:
            bridging_mode = "soft" if sinkhorn_use_soft_bridging else "hard"
            print(f"Sinkhorn bridging mode: {bridging_mode} (explicit)")
        else:
            print(f"Coupling mode: auto (soft if reg > 0.01)")
    print(f"Z reconstruction weight: {z_recon_weight}")
    print(f"No-sampling ratio: {no_sampling_ratio} (fraction using z_enc directly)")
    print(f"Regularization beta: {beta}")
    if bridging_method == 'nep':
        print(f"NEP weight: {nep_weight}")
        print(f"NEP var weight: {nep_var_weight}")
    if bridging_method == 'inv_softnn':
        print(f"Inverted SoftNN count_var_weight: {count_var_weight}")
    print(f"Dimension: {dim}")
    print("=" * 60)

    # Determine input dimension
    input_dim = 1 if dim == '1d' else 2

    # Generate data
    print(f"Generating {n_samples} synthetic data samples ({dim})...")
    if dim == '1d':
        x_data = generate_data(n_samples=n_samples, seed=seed)
        x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)  # (n_samples, 1)
    else:
        x_data = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
        x_data = torch.FloatTensor(x_data).to(device)  # (n_samples, 2)

    # Set initial sinkhorn_reg
    if sinkhorn_reg_schedule == 'fixed':
        init_reg = sinkhorn_reg
    else:
        init_reg = sinkhorn_reg_init
    
    # Create model
    model = NOVAE(
        input_dim=input_dim,
        latent_dim=input_dim,
        n_prior_samples=n_prior_resolved,
        bridging_method=bridging_method,
        sinkhorn_reg=init_reg,
        sinkhorn_use_soft_bridging=sinkhorn_use_soft_bridging,
        temperature=temperature_init if bridging_method in ['softnn', 'ot_guided_soft', 'inv_softnn'] else temperature,
        use_ste=use_ste if bridging_method in ['softnn', 'inv_softnn'] else False
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    if lr_scheduler_params is None:
        lr_scheduler_params = {}

    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
            **lr_scheduler_params
        )
    elif lr_scheduler == 'step':
        step_size = lr_scheduler_params.get('step_size', epochs // 3)
        gamma = lr_scheduler_params.get('gamma', 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'exponential':
        gamma = lr_scheduler_params.get('gamma', 0.995)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler == 'none' or lr_scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    # Import visualizations
    if dim == '1d':
        from visualization.viz_novae import visualize_novae_1d
    else:
        from visualization.viz_novae_2d import visualize_novae_2d_recon, visualize_novae_2d_gen

    os.makedirs(save_dir, exist_ok=True)

    # Fixed inference samples for consistent visualization across epochs
    n_infer = 200
    viz_z_infer = sample_prior(n_samples=n_infer, seed=seed + 1000, dim=input_dim)
    if dim == '1d':
        viz_x_data = generate_data(n_samples=n_samples, seed=seed)
    else:
        viz_x_data = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)

    # Training loop
    history = {'loss': []}
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Update sinkhorn_reg schedule (epsilon annealing) for Sinkhorn/NEP bridging
        if bridging_method in ['sinkhorn', 'nep']:
            if sinkhorn_reg_schedule == 'linear':
                # Linear decay: reg = init + (final - init) * (epoch / epochs)
                current_reg = sinkhorn_reg_init + (sinkhorn_reg_final - sinkhorn_reg_init) * (epoch / epochs)
            elif sinkhorn_reg_schedule == 'exponential':
                # Exponential decay: reg = init * (final/init)^(epoch/epochs)
                current_reg = sinkhorn_reg_init * (sinkhorn_reg_final / sinkhorn_reg_init) ** (epoch / epochs)
            elif sinkhorn_reg_schedule == 'cosine':
                # Cosine annealing: reg = final + (init - final) * (1 + cos(π * epoch/epochs)) / 2
                import math
                current_reg = sinkhorn_reg_final + (sinkhorn_reg_init - sinkhorn_reg_final) * \
                             (1 + math.cos(math.pi * epoch / epochs)) / 2
            else:  # 'fixed'
                current_reg = sinkhorn_reg
            
            model.set_sinkhorn_reg(current_reg)
        
        # Update temperature schedule for Soft NN, OT-Guided Soft, and Inverted SoftNN bridging
        if bridging_method in ['softnn', 'ot_guided_soft', 'inv_softnn']:
            if temperature_schedule == 'linear':
                current_temp = temperature_init + (temperature_final - temperature_init) * (epoch / epochs)
            elif temperature_schedule == 'exponential':
                current_temp = temperature_init * (temperature_final / temperature_init) ** (epoch / epochs)
            elif temperature_schedule == 'cosine':
                import math
                current_temp = temperature_final + (temperature_init - temperature_final) * \
                              (1 + math.cos(math.pi * epoch / epochs)) / 2
            else:  # 'fixed'
                current_temp = temperature
            
            model.set_temperature(current_temp)
        
        model.train()
        epoch_loss = 0.0

        # Shuffle data
        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            x_batch = x_shuffled[start_idx:end_idx]

            # Sample N prior samples for this batch (default: batch_size)
            n_prior = n_prior_resolved
            z_prior = torch.randn(n_prior, input_dim).to(device)

            # Forward pass: encode -> OT bridging -> decode
            no_sampling_val = no_sampling_ratio if no_sampling_ratio > 0 else None
            x_hat, z_, z_selected, weights, no_sampling_mask = model(
                x_batch, z_prior, no_sampling_ratio=no_sampling_val
            )

            # Loss: reconstruction + latent reconstruction + alignment + NEP
            matching_info = None
            nep_info = None
            inv_softnn_info = None
            if bridging_method == 'ot_guided_soft' and isinstance(weights, tuple):
                matching_info = weights  # (matching_indices, distances_sq)
            elif bridging_method == 'nep' and isinstance(weights, dict):
                nep_info = weights  # dict with 'assignments' and 'distances_sq'
            elif bridging_method == 'inv_softnn' and isinstance(weights, dict):
                inv_softnn_info = weights  # dict with 'counts', 'distances_sq', 'selected_indices'

            # Adjust z_recon_weight based on bridging method
            # softnn, ot_guided_soft, nep don't use z_recon loss; inv_softnn DOES use it (distance min)
            effective_z_recon_weight = 0.0 if bridging_method in ['softnn', 'ot_guided_soft', 'nep'] else z_recon_weight

            loss = model.loss_function(x_batch, x_hat, z_, z_selected,
                                     z_recon_weight=effective_z_recon_weight,
                                     alignment_weight=alignment_weight,
                                     matching_info=matching_info,
                                     no_sampling_mask=no_sampling_mask,
                                     beta=beta,
                                     nep_weight=nep_weight,
                                     nep_var_weight=nep_var_weight,
                                     nep_info=nep_info,
                                     inv_softnn_info=inv_softnn_info,
                                     count_var_weight=count_var_weight if bridging_method == 'inv_softnn' else 0.0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual

        # Average epoch loss
        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if bridging_method in ['sinkhorn', 'nep']:
                reg_str = f", Reg: {current_reg:.4f}" if sinkhorn_reg_schedule != 'fixed' else ""
            elif bridging_method in ['softnn', 'ot_guided_soft', 'inv_softnn']:
                reg_str = f", Temp: {current_temp:.4f}" if temperature_schedule != 'fixed' else ""
            else:
                reg_str = ""
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Loss: {epoch_loss:.6f}, "
                  f"LR: {current_lr:.6e}{reg_str}")

        # Generate visualization: every epoch for first 20 epochs, then every 10 epochs
        if epoch < 20:
            should_viz = True  # Every epoch for first 20 epochs
        else:
            should_viz = (epoch + 1) % 10 == 0  # Every 10 epochs after epoch 20
        
        if should_viz:
            model.eval()
            with torch.no_grad():
                # Compute reconstruction of training data
                if dim == '1d':
                    x_data_tensor = torch.FloatTensor(viz_x_data).unsqueeze(1).to(device)
                else:
                    x_data_tensor = torch.FloatTensor(viz_x_data).to(device)
                
                # Forward pass: encode -> soft NN -> decode
                # Reconstruction inference: n_prior_samples_recon defaults to input batch size (len(viz_x_data))
                n_prior_viz = n_prior_samples_recon if n_prior_samples_recon is not None else len(viz_x_data)
                z_prior_viz = torch.randn(n_prior_viz, input_dim).to(device)
                x_recon_tensor, _, _, _, _ = model(x_data_tensor, z_prior_viz)
                
                if dim == '1d':
                    x_recon = x_recon_tensor.squeeze().cpu().numpy()
                    z_tensor = torch.FloatTensor(viz_z_infer).unsqueeze(1).to(device)
                    x_infer = model.decode(z_tensor).squeeze().cpu().numpy()

                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_novae_1d(
                        z_infer=viz_z_infer,
                        x_infer=x_infer,
                        x_data=viz_x_data,
                        x_recon=x_recon,
                        save_path=viz_path,
                        epoch=epoch + 1
                    )
                else:
                    # Get full reconstruction pipeline outputs
                    # Reconstruction inference: n_prior_samples_recon defaults to input batch size (len(viz_x_data))
                    n_prior_viz = n_prior_samples_recon if n_prior_samples_recon is not None else len(viz_x_data)
                    z_prior_viz = torch.randn(n_prior_viz, input_dim).to(device)
                    x_recon_tensor, z_tensor, z_selected_tensor, _, _ = model(x_data_tensor, z_prior_viz)
                    
                    x_recon = x_recon_tensor.cpu().numpy()
                    z_ = z_tensor.cpu().numpy()
                    z_selected = z_selected_tensor.cpu().numpy()
                    
                    # Generation: z (prior) → decoder → x'
                    z_tensor_infer = torch.FloatTensor(viz_z_infer).to(device)
                    x_infer = model.decode(z_tensor_infer).cpu().numpy()

                    # Save reconstruction visualization
                    viz_path_recon = os.path.join(save_dir, f'epoch_{epoch+1:04d}_recon.png')
                    visualize_novae_2d_recon(
                        x_data=viz_x_data,
                        z_=z_,
                        z_selected=z_selected,
                        x_recon=x_recon,
                        save_path=viz_path_recon,
                        epoch=epoch + 1
                    )
                    
                    # Save generation visualization
                    viz_path_gen = os.path.join(save_dir, f'epoch_{epoch+1:04d}_gen.png')
                    visualize_novae_2d_gen(
                        z_infer=viz_z_infer,
                        x_infer=x_infer,
                        x_data=viz_x_data,
                        save_path=viz_path_gen,
                        epoch=epoch + 1
                    )
            model.train()

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    return model, history


def _parse_n_prior_samples(value):
    """Parse n_prior_samples: int, 'batch_size', 'n_data', or None."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ('none', ''):
        return None
    try:
        return int(s)
    except ValueError:
        return s.lower()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NO-VAE model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_prior_samples', type=_parse_n_prior_samples, default=None,
                        help='Prior samples: int, "batch_size", "n_data" (default: None, uses batch_size)')
    parser.add_argument('--n_prior_samples_recon', type=int, default=None,
                        help='Prior samples for recon inference (default: None, uses input batch size = n_samples)')
    parser.add_argument('--no_sampling_ratio', type=float, default=0.0,
                        help='Fraction of batch to use z_enc directly (no bridging) for reconstruction (default: 0.0)')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='Weight for regularization loss KL(z_enc batch || N(0,I)) (default: 0.0)')
    parser.add_argument('--nep_weight', type=float, default=0.0,
                        help='Weight for NEP loss (only when bridging_method=nep, default: 0.0)')
    parser.add_argument('--nep_var_weight', type=float, default=0.0,
                        help='Weight for NEP distance variance penalty (default: 0.0)')
    parser.add_argument('--bridging_method', type=str, default='sinkhorn',
                        choices=['sinkhorn', 'softnn', 'ot_guided_soft', 'nep', 'inv_softnn'],
                        help='Bridging method (default: sinkhorn)')
    parser.add_argument('--count_var_weight', type=float, default=0.1,
                        help='Inverted SoftNN load balancing weight (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for soft nearest neighbor (if schedule=fixed)')
    parser.add_argument('--temperature_schedule', type=str, default='cosine',
                        choices=['fixed', 'linear', 'exponential', 'cosine'],
                        help='Temperature schedule type (default: cosine)')
    parser.add_argument('--temperature_init', type=float, default=0.01,
                       help='Initial temperature for scheduling (default: 0.01)')
    parser.add_argument('--temperature_final', type=float, default=0.0,
                       help='Final temperature for scheduling (default: 0.0)')
    parser.add_argument('--dim', type=str, default='1d', choices=['1d', '2d'],
                        help='Dimension (1d or 2d)')
    parser.add_argument('--dataset_2d', type=str, default='2gauss',
                        choices=['2gauss', 'shifted_2gauss', 'two_moon'],
                        help='2D dataset type')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/Desktop/Gen_Study/outputs',
                        help='Directory to save model and visualization')
    parser.add_argument('--viz_freq', type=int, default=200,
                        help='Visualization frequency (every N epochs)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, history = train_novae(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        n_prior_samples=args.n_prior_samples,
        n_prior_samples_recon=args.n_prior_samples_recon,
        bridging_method=getattr(args, 'bridging_method', getattr(args, 'coupling_method', 'sinkhorn')),
        sinkhorn_reg=getattr(args, 'sinkhorn_reg', 0.05),
        sinkhorn_use_soft_bridging=getattr(args, 'sinkhorn_use_soft_bridging', getattr(args, 'sinkhorn_use_soft_coupling', False)),
        no_sampling_ratio=getattr(args, 'no_sampling_ratio', 0.0),
        beta=getattr(args, 'beta', 0.0),
        nep_weight=getattr(args, 'nep_weight', 0.0),
        nep_var_weight=getattr(args, 'nep_var_weight', 0.0),
        count_var_weight=getattr(args, 'count_var_weight', 0.1),
        viz_freq=args.viz_freq,
        save_dir=args.output_dir,
        dim=args.dim,
        dataset_2d=args.dataset_2d
    )

    # Save model
    save_path = os.path.join(args.output_dir, 'novae_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        input_dim = 1 if args.dim == '1d' else 2
        z_test = torch.randn(5, input_dim).to(device)
        x_gen = model.decode(z_test)
        print(f"z samples: {z_test.squeeze().cpu().numpy()}")
        print(f"Generated x: {x_gen.squeeze().cpu().numpy()}")
