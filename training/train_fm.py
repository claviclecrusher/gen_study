"""
Training script for Flow Matching model
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os
from scipy.stats import gaussian_kde

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_matching import FlowMatching
from data.synthetic import generate_data, generate_data_2d
from utils.cfm_sampler import create_cfm_sampler


def train_fm(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
             viz_freq=200, save_dir='/home/user/Desktop/Gen_Study/outputs', dim='1d',
             cfm_type='icfm', cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
             dataset_2d='2gauss', lr_scheduler='cosine', lr_scheduler_params=None):
    """
    Train Flow Matching model

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on
        viz_freq: Frequency to save visualizations (every N epochs)
        save_dir: Directory to save visualizations
        dim: Dimension ('1d' or '2d')
        cfm_type: CFM coupling type ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
        cfm_reg: Entropic regularization for Sinkhorn
        cfm_reg_m: Marginal regularization for unbalanced OT
        cfm_weight_power: Power factor for UOTRFM weights

    Returns:
        model: Trained flow matching model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training Flow Matching Model")
    print(f"CFM Type: {cfm_type.upper()}")
    print("=" * 60)

    # Create CFM sampler
    cfm_sampler = create_cfm_sampler(
        cfm_type=cfm_type,
        reg=cfm_reg,
        reg_m=cfm_reg_m,
        weight_power=cfm_weight_power
    )

    # Determine input dimension
    input_dim = 1 if dim == '1d' else 2

    # Generate data
    print(f"Generating {n_samples} synthetic data samples ({dim})...")
    if dim == '1d':
        x_data = generate_data(n_samples=n_samples, seed=seed)
        x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)
    else:  # 2d
        x_data = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
        x_data = torch.FloatTensor(x_data).to(device)

    # Create model
    model = FlowMatching(input_dim=input_dim).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    if lr_scheduler_params is None:
        lr_scheduler_params = {}
    
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01, **lr_scheduler_params)
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

    # Training loop
    history = {'loss': []}

    n_batches = (n_samples + batch_size - 1) // batch_size

    # Prepare visualization data
    if dim == '1d':
        from visualization.viz_fm import visualize_fm
    else:
        from visualization.viz_fm_2d import visualize_fm_2d

    from data.synthetic import sample_prior

    os.makedirs(save_dir, exist_ok=True)

    z_train_viz = sample_prior(n_samples=n_samples, seed=seed, dim=input_dim)
    if dim == '1d':
        x_train_viz = generate_data(n_samples=n_samples, seed=seed)
    else:
        x_train_viz = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
    coupling_indices_viz = np.random.permutation(n_samples)

    n_infer = 200
    z_infer_viz = sample_prior(n_samples=n_infer, seed=seed + 1000, dim=input_dim)

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
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
            z_batch = torch.randn(batch_size_actual, input_dim).to(device)
            t_batch = torch.rand(batch_size_actual, 1).to(device)

            # Apply CFM coupling
            z_coupled, x_coupled, weights = cfm_sampler.sample_coupling(z_batch, x_batch)

            optimizer.zero_grad()
            loss = model.loss_function(z_coupled, x_coupled, t_batch, weights=weights)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual

        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6e}")

        # Generate visualization every viz_freq epochs
        if (epoch + 1) % viz_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                if dim == '1d':
                    # Generate ODE trajectories
                    z_tensor = torch.FloatTensor(z_infer_viz).unsqueeze(1).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.squeeze().cpu().numpy()

                    # Save visualization
                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_fm(
                        z_samples=z_train_viz,
                        x_data=x_train_viz,
                        trajectories=trajectories,
                        coupling_indices=coupling_indices_viz,
                        save_path=viz_path,
                        vector_info=None
                    )
                else:  # 2d
                    # Generate trajectories
                    z_tensor = torch.FloatTensor(z_infer_viz).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.cpu().numpy()  # (n_steps+1, n_infer, 2)

                    # Save visualization
                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_fm_2d(
                        trajectories=trajectories,
                        x_data=x_train_viz,
                        save_path=viz_path,
                        epoch=epoch + 1,
                        cfm_type=cfm_type
                    )
            model.train()

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    return model, history


if __name__ == '__main__':
    import argparse
    from visualization.viz_fm import visualize_fm
    from data.synthetic import sample_prior

    parser = argparse.ArgumentParser(description='Train Flow Matching model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Gen_Study/outputs',
                        help='Directory to save model and visualization')
    parser.add_argument('--viz_freq', type=int, default=200,
                        help='Frequency to save visualizations (every N epochs)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train model
    model, history = train_fm(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        viz_freq=args.viz_freq,
        save_dir=args.output_dir
    )

    save_path = os.path.join(args.output_dir, 'fm_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        z_test = torch.randn(10, 1).to(device)
        trajectory = model.sample(z_test, n_steps=100, device=device)
        print(f"Sample trajectory shape: {trajectory.shape}")
        print(f"Final samples: {trajectory[-1].squeeze().cpu().numpy()}")

    # Create visualization
    print("\nCreating visualization...")
    z_train = sample_prior(n_samples=args.n_samples, seed=args.seed)
    x_train = generate_data(n_samples=args.n_samples, seed=args.seed)
    coupling_indices = np.random.permutation(args.n_samples)

    n_infer = 200
    z_infer = sample_prior(n_samples=n_infer, seed=args.seed + 1)
    with torch.no_grad():
        trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
        trajectories = trajectories_tensor.squeeze().cpu().numpy()

    viz_path = os.path.join(args.output_dir, 'fm_visualization.png')

    visualize_fm(
        z_samples=z_train,
        x_data=x_train,
        trajectories=trajectories,
        coupling_indices=coupling_indices,
        save_path=viz_path,
        vector_info=None
    )

    print("\nVisualization saved!")
