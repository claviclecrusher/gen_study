"""
Training script for Non-identifiable Decoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.decoder import Decoder
from data.synthetic import generate_data, sample_prior


def train_decoder(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu'):
    """
    Train Non-identifiable Decoder model

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on

    Returns:
        model: Trained decoder model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training Non-identifiable Decoder")
    print("=" * 60)

    # Generate data
    print(f"Generating {n_samples} synthetic data samples...")
    x_data = generate_data(n_samples=n_samples, seed=seed)
    x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)  # (n_samples, 1)

    # Create model
    model = Decoder(latent_dim=1, output_dim=1).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    history = {'loss': []}

    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Shuffle data for random coupling
        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]

        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            # Sample random noise z ~ N(0, I)
            z = torch.randn(batch_size_actual, 1).to(device)

            # Get target x (random coupling)
            x_target = x_shuffled[start_idx:end_idx]

            # Forward pass
            x_hat = model(z)

            # Compute loss
            loss = criterion(x_hat, x_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual

        # Average loss
        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)

        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Decoder model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Gen_Study/outputs',
                        help='Directory to save model and visualization')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, history = train_decoder(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device
    )

    # Save model
    save_path = os.path.join(args.output_dir, 'nid_decoder_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        z_test = torch.randn(10, 1).to(device)
        x_test = model(z_test)
        print(f"Sample outputs: {x_test.squeeze().cpu().numpy()}")
        print(f"Mean of outputs: {x_test.mean().item():.4f}")
