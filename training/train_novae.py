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


def train_novae(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
               n_prior_samples=None, coupling_method='sinkhorn', sinkhorn_reg=0.05,
               sinkhorn_reg_schedule='cosine', sinkhorn_reg_init=1.0, sinkhorn_reg_final=0.01,
               sinkhorn_use_soft_coupling=False, z_recon_weight=1.0,
               temperature=0.1, temperature_schedule='cosine', temperature_init=0.01, temperature_final=0.0,
               use_ste=False, alignment_weight=1.0,
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
        n_prior_samples: Number of prior samples N per mini-batch (default: None, uses n_samples)
        coupling_method: Coupling method ('sinkhorn', default)
        sinkhorn_reg: Entropic regularization for Sinkhorn (if schedule='fixed', default: 0.05)
        sinkhorn_reg_schedule: Epsilon annealing schedule ('fixed', 'linear', 'exponential', 'cosine', default: 'cosine')
        sinkhorn_reg_init: Initial epsilon for annealing (default: 1.0)
        sinkhorn_reg_final: Final epsilon for annealing (default: 0.01)
        sinkhorn_use_soft_coupling: If True, always use soft coupling. If False, always use hard coupling.
                                    If None, auto-select based on sinkhorn_reg (default: None)
        z_recon_weight: Weight for latent reconstruction loss (default: 1.0)
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

    print("=" * 60)
    print(f"Training Noise Oriented VAE (NO-VAE)")
    n_prior_display = n_prior_samples if n_prior_samples is not None else f"n_samples ({n_samples})"
    print(f"N prior samples: {n_prior_display}")
    print(f"Coupling method: {coupling_method}")
    if coupling_method == 'sinkhorn':
        if sinkhorn_reg_schedule == 'fixed':
            print(f"Sinkhorn regularization: {sinkhorn_reg} (fixed)")
        else:
            print(f"Sinkhorn regularization schedule: {sinkhorn_reg_schedule}")
            print(f"  Initial: {sinkhorn_reg_init}, Final: {sinkhorn_reg_final}")
        if sinkhorn_use_soft_coupling is not None:
            coupling_mode = "soft" if sinkhorn_use_soft_coupling else "hard"
            print(f"Sinkhorn coupling mode: {coupling_mode} (explicit)")
        else:
            print(f"Coupling mode: auto (soft if reg > 0.01)")
    print(f"Z reconstruction weight: {z_recon_weight}")
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
        n_prior_samples=n_prior_samples,
        coupling_method=coupling_method,
        sinkhorn_reg=init_reg,
        sinkhorn_use_soft_coupling=sinkhorn_use_soft_coupling,
        temperature=temperature_init if coupling_method in ['softnn', 'ot_guided_soft'] else temperature,
        use_ste=use_ste if coupling_method == 'softnn' else False
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
        # Update sinkhorn_reg schedule (epsilon annealing) for Sinkhorn coupling
        if coupling_method == 'sinkhorn':
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
        
        # Update temperature schedule for Soft NN and OT-Guided Soft coupling
        if coupling_method in ['softnn', 'ot_guided_soft']:
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

            # Sample N prior samples for this batch
            # Use n_samples if n_prior_samples is None
            n_prior = n_prior_samples if n_prior_samples is not None else n_samples
            z_prior = torch.randn(n_prior, input_dim).to(device)

            # Forward pass: encode -> OT coupling -> decode
            x_hat, z_, z_selected, weights = model(x_batch, z_prior)

            # Loss: reconstruction + latent reconstruction + alignment (for ot_guided_soft)
            # For softnn and ot_guided_soft: z_recon_weight should be 0 (only reconstruction loss)
            matching_info = None
            if coupling_method == 'ot_guided_soft' and isinstance(weights, tuple):
                matching_info = weights  # (matching_indices, distances_sq)
            
            # Adjust z_recon_weight based on coupling method
            # softnn and ot_guided_soft don't use z_recon loss (only reconstruction + alignment)
            effective_z_recon_weight = 0.0 if coupling_method in ['softnn', 'ot_guided_soft'] else z_recon_weight
            
            loss = model.loss_function(x_batch, x_hat, z_, z_selected, 
                                     z_recon_weight=effective_z_recon_weight,
                                     alignment_weight=alignment_weight,
                                     matching_info=matching_info)

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
            if coupling_method == 'sinkhorn':
                reg_str = f", Reg: {current_reg:.4f}" if sinkhorn_reg_schedule != 'fixed' else ""
            elif coupling_method in ['softnn', 'ot_guided_soft']:
                reg_str = f", Temp: {current_temp:.4f}" if temperature_schedule != 'fixed' else ""
            else:
                reg_str = ""
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Loss: {epoch_loss:.6f}, "
                  f"LR: {current_lr:.6e}{reg_str}")

        # Generate visualization every viz_freq epochs
        if (epoch + 1) % viz_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Compute reconstruction of training data
                if dim == '1d':
                    x_data_tensor = torch.FloatTensor(viz_x_data).unsqueeze(1).to(device)
                else:
                    x_data_tensor = torch.FloatTensor(viz_x_data).to(device)
                
                # Forward pass: encode -> soft NN -> decode
                # Use viz_x_data size if n_prior_samples is None
                n_prior_viz = n_prior_samples if n_prior_samples is not None else len(viz_x_data)
                z_prior_viz = torch.randn(n_prior_viz, input_dim).to(device)
                x_recon_tensor, _, _, _ = model(x_data_tensor, z_prior_viz)
                
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
                    # Use viz_x_data size if n_prior_samples is None
                    n_prior_viz = n_prior_samples if n_prior_samples is not None else len(viz_x_data)
                    z_prior_viz = torch.randn(n_prior_viz, input_dim).to(device)
                    x_recon_tensor, z_tensor, z_selected_tensor, _ = model(x_data_tensor, z_prior_viz)
                    
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train NO-VAE model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_prior_samples', type=int, default=1024,
                        help='Number of prior samples per batch')
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
        coupling_method=args.coupling_method,
        sinkhorn_reg=args.sinkhorn_reg,
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
