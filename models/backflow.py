"""
BackFlow model for 1D data
Adapted from original CIFAR10 implementation
"""
import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BackFlow(nn.Module):
    """
    BackFlow model for 1D data.

    The model takes three inputs:
    - x: data point at time t (shape: [B, 1])
    - r: reference time (shape: [B])
    - t: current time (shape: [B])

    Returns:
    - u: mean velocity field u(x, r, t)
    """
    def __init__(self, input_dim=1, hidden_dim=256, time_emb_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding for both t and (t-r)
        self.pos_emb = SinusoidalPosEmb(time_emb_dim // 2)

        # Time MLP to process concatenated [t_emb, tr_emb]
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Main network: concatenate x and time embedding
        self.network = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize last layer to zero (like zero_module in original)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x, r, t):
        """
        Forward pass.

        Args:
            x: Input data [B, 1]
            r: Reference time [B]
            t: Current time [B]

        Returns:
            u: Mean velocity field [B, 1]
        """
        # Compute time embeddings
        t_emb = self.pos_emb(t)  # [B, time_emb_dim/2]
        tr_emb = self.pos_emb(t - r)  # [B, time_emb_dim/2]

        # Concatenate and process time embeddings
        cond = torch.cat([t_emb, tr_emb], dim=-1)  # [B, time_emb_dim]
        cond = self.time_mlp(cond)  # [B, time_emb_dim]

        # Concatenate x and time embedding
        h = torch.cat([x, cond], dim=-1)  # [B, input_dim + time_emb_dim]

        # Main network
        u = self.network(h)  # [B, input_dim]

        return u
