"""
Training script for Improved MeanFlow (iMF) model
Reference: https://arxiv.org/html/2512.02012v1
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.improved_mean_flow import ImprovedMeanFlow
from data.synthetic import generate_data, generate_data_2d, sample_prior
from utils.cfm_sampler import create_cfm_sampler


def train_imf(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
              viz_interval=None, viz_output_dir=None, flow_ratio=0.5, n_infer=200, dim='1d',
              cfm_type='icfm', cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
              dataset_2d='2gauss', lr_scheduler='cosine', lr_scheduler_params=None):
    """
    Train Improved MeanFlow (iMF) model

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on
        viz_interval: Interval (in epochs) to save visualization. If None, no intermediate visualizations.
        viz_output_dir: Directory to save intermediate visualizations. Required if viz_interval is set.
        flow_ratio: Ratio of samples where r=t (default: 0.5)
        n_infer: Number of inference samples for visualization (default: 200)
        dim: Dimension ('1d' or '2d')
        cfm_type: CFM coupling type ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
        cfm_reg: Entropic regularization for Sinkhorn
        cfm_reg_m: Marginal regularization for unbalanced OT
        cfm_weight_power: Power factor for UOTRFM weights

    Returns:
        model: Trained Improved MeanFlow model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training Improved MeanFlow (iMF) Model")
    print(f"CFM Type: {cfm_type.upper()}")
    print("=" * 60)
    print("Key improvement: v-loss reformulation with ground-truth target")

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
    model = ImprovedMeanFlow(input_dim=input_dim).to(device)
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

    # Prepare visualization data if needed
    if viz_interval is not None and viz_output_dir is not None:
        if dim == '1d':
            from visualization.viz_imf import visualize_imf
        else:
            from visualization.viz_imf_2d import visualize_imf_2d

        os.makedirs(viz_output_dir, exist_ok=True)

        # Generate training and inference data for visualization
        z_train_viz = sample_prior(n_samples=n_samples, seed=seed, dim=input_dim)
        if dim == '1d':
            x_train_viz = generate_data(n_samples=n_samples, seed=seed)
        else:
            x_train_viz = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
        coupling_indices_viz = np.random.permutation(n_samples)

        z_infer_viz = sample_prior(n_samples=n_infer, seed=seed + 1, dim=input_dim)

    # Training loop
    history = {'loss': []}

    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nTraining for {epochs} epochs...")
    if viz_interval is not None:
        print(f"Saving visualization every {viz_interval} epochs to {viz_output_dir}")

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

            # Sample noise for CFM coupling
            e_batch = torch.randn(batch_size_actual, input_dim).to(device)

            # Apply CFM coupling (e=noise, x=data)
            e_coupled, x_coupled, weights = cfm_sampler.sample_coupling(e_batch, x_batch)

            # Sample t and r following the same strategy as original MeanFlow
            # Generate two random samples
            s1 = torch.rand(batch_size_actual, 1).to(device)
            s2 = torch.rand(batch_size_actual, 1).to(device)

            # t = max(s1, s2), r = min(s1, s2)
            t_batch = torch.max(s1, s2)
            r_batch = torch.min(s1, s2)

            # Apply flow_ratio: set r=t for flow_ratio portion of the batch
            num_flow = int(flow_ratio * batch_size_actual)
            if num_flow > 0:
                flow_indices = torch.randperm(batch_size_actual)[:num_flow]
                r_batch[flow_indices] = t_batch[flow_indices]

            optimizer.zero_grad()
            loss = model.loss_function(x_coupled, t_batch, r_batch, e=e_coupled, weights=weights)
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

        # Save intermediate visualization
        if viz_interval is not None and viz_output_dir is not None:
            if (epoch + 1) % viz_interval == 0:
                print(f"  Saving visualization at epoch {epoch+1}...")
                model.eval()
                with torch.no_grad():
                    if dim == '1d':
                        z_tensor = torch.FloatTensor(z_infer_viz).unsqueeze(1).to(device)

                        # ODE trajectories
                        trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                        trajectories = trajectories_tensor.squeeze().cpu().numpy()

                        # Mean velocity ODE trajectories (2 steps)
                        mean_trajectories_tensor = model.sample_mean_velocity_ode(z_tensor, n_steps=2, device=device)
                        mean_trajectories = mean_trajectories_tensor.squeeze().cpu().numpy()

                        # One-step predictions (mean velocity)
                        mean_predictions_tensor = model.sample_mean_velocity(z_tensor, device=device)
                        mean_predictions = mean_predictions_tensor.squeeze().cpu().numpy()

                        viz_path = os.path.join(viz_output_dir, f'epoch_{epoch+1:04d}.png')
                        visualize_imf(
                            z_samples=z_train_viz,
                            x_data=x_train_viz,
                            trajectories=trajectories,
                            mean_predictions=mean_predictions,
                            coupling_indices=coupling_indices_viz,
                            save_path=viz_path,
                            mean_trajectories=mean_trajectories
                        )
                    else:  # 2d
                        z_tensor = torch.FloatTensor(z_infer_viz).to(device)

                        # ODE trajectories
                        trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                        trajectories = trajectories_tensor.cpu().numpy()

                        # One-step prediction (mean velocity)
                        mean_onestep_tensor = model.sample_mean_velocity(z_tensor, device=device)
                        mean_onestep = mean_onestep_tensor.cpu().numpy()

                        viz_path = os.path.join(viz_output_dir, f'epoch_{epoch+1:04d}.png')
                        visualize_imf_2d(
                            trajectories=trajectories,
                            mean_onestep=mean_onestep,
                            z_samples=z_infer_viz,
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
    from visualization.viz_imf import visualize_imf
    from data.synthetic import sample_prior

    parser = argparse.ArgumentParser(description='Train Improved MeanFlow (iMF) model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Gen_Study/outputs',
                        help='Directory to save model and visualization')
    parser.add_argument('--viz_interval', type=int, default=None,
                        help='Save visualization every N epochs (default: None, only save final)')
    parser.add_argument('--flow_ratio', type=float, default=0.5,
                        help='Ratio of samples where r=t (default: 0.5)')
    parser.add_argument('--n_infer', type=int, default=200,
                        help='Number of inference samples for visualization (default: 200)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train model
    model, history = train_imf(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        viz_interval=args.viz_interval,
        viz_output_dir=args.output_dir if args.viz_interval else None,
        flow_ratio=args.flow_ratio,
        n_infer=args.n_infer
    )

    save_path = os.path.join(args.output_dir, 'imf_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nTesting inference...")
    model.eval()

    # Test one-step sampling (mean velocity)
    z_test = torch.randn(10, 1).to(device)
    with torch.no_grad():
        x_pred_mean = model.sample_mean_velocity(z_test, device=device)
        print(f"One-step samples (mean velocity): {x_pred_mean.squeeze().cpu().numpy()}")

    # Test multi-step sampling (ODE trajectory)
    with torch.no_grad():
        trajectory = model.sample(z_test, n_steps=100, device=device)
        print(f"Sample trajectory shape: {trajectory.shape}")
        print(f"Final samples (ODE): {trajectory[-1].squeeze().cpu().numpy()}")

    # Create visualization
    print("\nCreating visualization...")
    z_train = sample_prior(n_samples=args.n_samples, seed=args.seed)
    x_train = generate_data(n_samples=args.n_samples, seed=args.seed)
    coupling_indices = np.random.permutation(args.n_samples)

    z_infer = sample_prior(n_samples=args.n_infer, seed=args.seed + 1)

    with torch.no_grad():
        # ODE trajectories
        trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
        trajectories = trajectories_tensor.squeeze().cpu().numpy()

        # Mean velocity ODE trajectories (2 steps)
        mean_trajectories_tensor = model.sample_mean_velocity_ode(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=2, device=device)
        mean_trajectories = mean_trajectories_tensor.squeeze().cpu().numpy()

        # One-step predictions (mean velocity)
        mean_predictions_tensor = model.sample_mean_velocity(torch.FloatTensor(z_infer).unsqueeze(1).to(device), device=device)
        mean_predictions = mean_predictions_tensor.squeeze().cpu().numpy()

    viz_path = os.path.join(args.output_dir, 'imf_visualization.png')
    visualize_imf(
        z_samples=z_train,
        x_data=x_train,
        trajectories=trajectories,
        mean_predictions=mean_predictions,
        coupling_indices=coupling_indices,
        save_path=viz_path,
        mean_trajectories=mean_trajectories
    )

    print("\nVisualization saved!")
