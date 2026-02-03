"""
Variational Autoencoder (VAE) model
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class VAE(nn.Module):
    """
    Variational Autoencoder model

    Training:
        Input: x (data sample)
        Encoding: x → (μ, log_σ²) → ẑ (reparameterization)
        Decoding: ẑ → x̂
        Loss: MSE(x̂, x) + KL(q(ẑ|x) || p(z))

    Args:
        input_dim: Dimension of input space (default: 1)
        latent_dim: Dimension of latent space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, input_dim=1, latent_dim=1, hidden_dims=None, beta=1.0):
        super(VAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight

        # Encoder network - maps to latent representation
        # Reverse hidden dims for encoder (symmetric architecture)
        encoder_hidden = list(reversed(hidden_dims))

        # Shared encoder backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.encoder_backbone = nn.Sequential(*layers)

        # Separate heads for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder network (inherits from BaseMLP)
        self.decoder = BaseMLP(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )

    def encode(self, x):
        """
        Encode input to latent distribution parameters

        Args:
            x: Input of shape (batch_size, input_dim)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)

        Args:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)

        Returns:
            z: Sampled latent code (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

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
        Forward pass through VAE

        Args:
            x: Input of shape (batch_size, input_dim)

        Returns:
            x_hat: Reconstructed output of shape (batch_size, input_dim)
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
            z_hat: Sampled latent code (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z_hat = self.reparameterize(mu, logvar)
        x_hat = self.decode(z_hat)
        return x_hat, mu, logvar, z_hat

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

    def loss_function(self, x, x_hat, mu, logvar):
        """
        Compute VAE loss = Reconstruction loss + beta * KL divergence

        Args:
            x: Original input (batch_size, input_dim)
            x_hat: Reconstructed output (batch_size, input_dim)
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)

        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss (MSE)
            kl_loss: KL divergence loss (unweighted)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')

        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch

        # Total loss with beta weighting
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Test VAE
    print("Testing VAE...")

    # Create model
    model = VAE(input_dim=1, latent_dim=1)
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1)
    x_hat, mu, logvar, z_hat = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Latent shape: {z_hat.shape}")
    print(f"Output shape: {x_hat.shape}")

    # Test loss computation
    loss, recon_loss, kl_loss = model.loss_function(x, x_hat, mu, logvar)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"\nSample shape: {samples.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")
