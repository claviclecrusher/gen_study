"""
Training script for Autoencoder
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import Autoencoder
from data.synthetic import generate_data


def train_autoencoder(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu'):
    """
    Train Autoencoder model

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on

    Returns:
        model: Trained autoencoder model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training Autoencoder")
    print("=" * 60)

    # Generate data
    print(f"Generating {n_samples} synthetic data samples...")
    x_data = generate_data(n_samples=n_samples, seed=seed)
    x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)  # (n_samples, 1)

    # Create model
    model = Autoencoder(input_dim=1, latent_dim=1).to(device)
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

        # Shuffle data
        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]

        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            x_batch = x_shuffled[start_idx:end_idx]

            # Forward pass
            x_hat, z_hat = model(x_batch)

            # Compute loss (reconstruction loss)
            loss = criterion(x_hat, x_batch)

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

    parser = argparse.ArgumentParser(description='Train Autoencoder model')
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

    model, history = train_autoencoder(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device
    )

    # Save model
    save_path = os.path.join(args.output_dir, 'autoencoder_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(5, 1).to(device)
        x_recon, z_encoded = model(x_test)
        print(f"Input samples: {x_test.squeeze().cpu().numpy()}")
        print(f"Encoded z: {z_encoded.squeeze().cpu().numpy()}")
        print(f"Reconstructed samples: {x_recon.squeeze().cpu().numpy()}")
