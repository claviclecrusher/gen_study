"""
Autoencoder model
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class Autoencoder(nn.Module):
    """
    Autoencoder model

    Training:
        Input: x (data sample)
        Encoding: x → ẑ
        Decoding: ẑ → x̂
        Loss: MSE(x̂, x)

    Args:
        input_dim: Dimension of input space (default: 1)
        latent_dim: Dimension of latent space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, input_dim=1, latent_dim=1, hidden_dims=None):
        super(Autoencoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder network (MLP)
        # Reverse hidden dims for encoder (symmetric architecture)
        encoder_hidden = list(reversed(hidden_dims))
        self.encoder = BaseMLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=encoder_hidden
        )

        # Decoder network (inherits from BaseMLP)
        self.decoder = BaseMLP(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )

    def encode(self, x):
        """
        Encode input to latent space

        Args:
            x: Input of shape (batch_size, input_dim)

        Returns:
            z_hat: Latent code of shape (batch_size, latent_dim)
        """
        z_hat = self.encoder(x)
        return z_hat

    def decode(self, z):
        """
        Decode latent code to output space

        Args:
            z: Latent code of shape (batch_size, latent_dim)

        Returns:
            x_hat: Reconstructed output of shape (batch_size, input_dim)
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        """
        Forward pass through autoencoder

        Args:
            x: Input of shape (batch_size, input_dim)

        Returns:
            x_hat: Reconstructed output of shape (batch_size, input_dim)
            z_hat: Latent code of shape (batch_size, latent_dim)
        """
        z_hat = self.encode(x)
        x_hat = self.decode(z_hat)
        return x_hat, z_hat

    def sample(self, n_samples, device='cpu'):
        """
        Sample from the model by passing random noise through decoder

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            x_hat: Generated samples of shape (n_samples, input_dim)
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        with torch.no_grad():
            x_hat = self.decode(z)
        return x_hat


if __name__ == "__main__":
    # Test Autoencoder
    print("Testing Autoencoder...")

    # Create model
    model = Autoencoder(input_dim=1, latent_dim=1)
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1)
    x_hat, z_hat = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Latent shape: {z_hat.shape}")
    print(f"Output shape: {x_hat.shape}")

    # Test encoding and decoding separately
    z = model.encode(x)
    x_recon = model.decode(z)
    print(f"\nEncoded shape: {z.shape}")
    print(f"Decoded shape: {x_recon.shape}")

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"Sample shape: {samples.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")
