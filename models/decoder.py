"""
Non-identifiable Decoder model
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class Decoder(nn.Module):
    """
    Non-identifiable Decoder model

    Training:
        Input: z ~ N(0, I) (random noise)
        Output: x̂ (reconstructed data)
        Loss: MSE(x̂, x) with random coupling between z and x

    Args:
        latent_dim: Dimension of latent space (default: 1)
        output_dim: Dimension of output space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, latent_dim=1, output_dim=1, hidden_dims=None):
        super(Decoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Decoder network (inherits from BaseMLP)
        self.decoder = BaseMLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )

    def forward(self, z):
        """
        Forward pass through decoder

        Args:
            z: Latent code of shape (batch_size, latent_dim)

        Returns:
            x_hat: Reconstructed output of shape (batch_size, output_dim)
        """
        x_hat = self.decoder(z)
        return x_hat

    def sample(self, n_samples, device='cpu'):
        """
        Sample from the model by passing random noise through decoder

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            x_hat: Generated samples of shape (n_samples, output_dim)
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        with torch.no_grad():
            x_hat = self.forward(z)
        return x_hat


if __name__ == "__main__":
    # Test Decoder
    print("Testing Decoder...")

    # Create model
    model = Decoder(latent_dim=1, output_dim=1)
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    z = torch.randn(batch_size, 1)
    x_hat = model(z)
    print(f"\nLatent input shape: {z.shape}")
    print(f"Output shape: {x_hat.shape}")

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"Sample shape: {samples.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")
