"""
Training script for 2D MeanFlow model
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mean_flow import MeanFlow
from data.synthetic_2d import generate_data_2d, sample_prior_2d


def train_meanflow_2d(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
                      viz_interval=None, viz_output_dir=None, flow_ratio=0.5, n_infer=200):
    """
    Train 2D MeanFlow model

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
        n_infer: Number of inference samples for visualization

    Returns:
        model: Trained MeanFlow model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training 2D MeanFlow Model")
    print("=" * 60)

    # Generate data
    print(f"Generating {n_samples} 2D synthetic data samples...")
    x_data = generate_data_2d(n_samples=n_samples, seed=seed)
    x_data = torch.FloatTensor(x_data).to(device)

    # Create model (2D input)
    model = MeanFlow(input_dim=2).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare visualization data if needed
    if viz_interval is not None and viz_output_dir is not None:
        from visualization.viz_meanflow_2d import visualize_meanflow_2d
        os.makedirs(viz_output_dir, exist_ok=True)

        # Generate training and inference data for visualization
        z_train_viz = sample_prior_2d(n_samples=n_samples, seed=seed)
        x_train_viz = generate_data_2d(n_samples=n_samples, seed=seed)
        coupling_indices_viz = np.random.permutation(n_samples)

        z_infer_viz = sample_prior_2d(n_samples=n_infer, seed=seed + 1)

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

            # Sample t and r following GitHub implementation
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
            loss = model.loss_function(x_batch, t_batch, r_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual

        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

        # Save intermediate visualization
        if viz_interval is not None and viz_output_dir is not None:
            if (epoch + 1) % viz_interval == 0:
                print(f"  Saving visualization at epoch {epoch+1}...")
                model.eval()
                with torch.no_grad():
                    # ODE trajectories (instantaneous velocity v)
                    trajectories_tensor = model.sample(
                        torch.FloatTensor(z_infer_viz).to(device),
                        n_steps=100,
                        device=device
                    )
                    trajectories = trajectories_tensor.cpu().numpy()

                    # Mean velocity ODE trajectories (u multi-step, 2 steps)
                    mean_trajectories_tensor = model.sample_mean_velocity_ode(
                        torch.FloatTensor(z_infer_viz).to(device),
                        n_steps=2,
                        device=device
                    )
                    mean_trajectories = mean_trajectories_tensor.cpu().numpy()

                    # One-step predictions (mean velocity u)
                    mean_predictions_tensor = model.sample_mean_velocity(
                        torch.FloatTensor(z_infer_viz).to(device),
                        device=device
                    )
                    mean_predictions = mean_predictions_tensor.cpu().numpy()

                viz_path = os.path.join(viz_output_dir, f'meanflow_2d_epoch{epoch+1:04d}.png')
                visualize_meanflow_2d(
                    z_samples=z_train_viz,
                    x_data=x_train_viz,
                    trajectories=trajectories,
                    mean_predictions=mean_predictions,
                    coupling_indices=coupling_indices_viz,
                    save_path=viz_path,
                    mean_trajectories=mean_trajectories
                )
                model.train()

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    return model, history


if __name__ == '__main__':
    import argparse
    from visualization.viz_meanflow_2d import visualize_meanflow_2d
    from data.synthetic_2d import sample_prior_2d

    parser = argparse.ArgumentParser(description='Train 2D MeanFlow model')
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
                        help='Number of inference samples for visualization')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train model
    model, history = train_meanflow_2d(
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

    save_path = os.path.join(args.output_dir, 'meanflow_2d_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    print("\nTesting inference...")
    model.eval()

    # Test one-step sampling (mean velocity)
    z_test = torch.randn(10, 2).to(device)
    with torch.no_grad():
        x_pred_mean = model.sample_mean_velocity(z_test, device=device)
        print(f"One-step samples (mean velocity):\n{x_pred_mean.cpu().numpy()}")

    # Test multi-step sampling (ODE trajectory)
    with torch.no_grad():
        trajectory = model.sample(z_test, n_steps=100, device=device)
        print(f"Sample trajectory shape: {trajectory.shape}")
        print(f"Final samples (ODE):\n{trajectory[-1].cpu().numpy()}")

    # Create visualization
    print("\nCreating visualization...")
    z_train = sample_prior_2d(n_samples=args.n_samples, seed=args.seed)
    x_train = generate_data_2d(n_samples=args.n_samples, seed=args.seed)
    coupling_indices = np.random.permutation(args.n_samples)

    z_infer = sample_prior_2d(n_samples=args.n_infer, seed=args.seed + 1)

    with torch.no_grad():
        # ODE trajectories (instantaneous velocity v)
        trajectories_tensor = model.sample(torch.FloatTensor(z_infer).to(device), n_steps=100, device=device)
        trajectories = trajectories_tensor.cpu().numpy()

        # Mean velocity ODE trajectories (u multi-step, 2 steps)
        mean_trajectories_tensor = model.sample_mean_velocity_ode(torch.FloatTensor(z_infer).to(device), n_steps=2, device=device)
        mean_trajectories = mean_trajectories_tensor.cpu().numpy()

        # One-step predictions (mean velocity u)
        mean_predictions_tensor = model.sample_mean_velocity(torch.FloatTensor(z_infer).to(device), device=device)
        mean_predictions = mean_predictions_tensor.cpu().numpy()

    viz_path = os.path.join(args.output_dir, 'meanflow_2d_visualization.png')
    visualize_meanflow_2d(
        z_samples=z_train,
        x_data=x_train,
        trajectories=trajectories,
        mean_predictions=mean_predictions,
        coupling_indices=coupling_indices,
        save_path=viz_path,
        mean_trajectories=mean_trajectories
    )

    print("\nVisualization saved!")
