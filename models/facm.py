"""
Flow-Anchored Consistency Models (FACM) for low-dimensional synthetic experiments.

This is a lightweight adaptation of the official FACM implementation to the
Gen_Study codebase (1D/2D synthetic data, MLP backbone, matplotlib visualization).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_mlp import BaseMLP


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    if x.ndim <= 1:
        return x
    return x.mean(dim=tuple(range(1, x.ndim)))


@dataclass
class FACMConfig:
    # Loss hyperparameters (from FACM paper/code style)
    p: float = 0.5
    c: float = 1e-3
    use_cosine_in_fm: bool = True
    clamp_g: float = 1.0

    # Time sampling
    t_type: str = "uniform"  # {"uniform", "log", "default"}
    mean: float = 0.0  # used by "default"/"log" sampler
    std: float = 1.0   # used by "default"/"log" sampler


class FACM(nn.Module):
    """
    FACM model with a MeanFlow-style backbone: f_theta(x, t, r).

    - For FM anchor loss we use r=t (as in FACM code).
    - For CM loss and CM sampler we use r=1 (end time).
    """

    def __init__(self, input_dim: int = 1, hidden_dims: Optional[list[int]] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim
        self.net = BaseMLP(
            input_dim=input_dim + 2,  # x + t + r
            output_dim=input_dim,
            hidden_dims=hidden_dims,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D)
            t: (B, 1)
            r: (B, 1)
        Returns:
            f: (B, D)
        """
        inp = torch.cat([x, t, r], dim=-1)
        return self.net(inp)

    # -------------------------
    # Time sampling (adapted)
    # -------------------------
    @staticmethod
    def sample_t(
        batch_size: int,
        device: torch.device,
        t_type: str = "uniform",
        mean: float = 0.0,
        std: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Returns:
            t in [0, 1], shape (B, 1)

        Notes:
        - "default"/"log" are included to mirror FACM repo options.
        - For low-dim synthetic tasks, "uniform" is usually sufficient.
        """
        if t_type == "uniform":
            t = torch.rand(batch_size, 1, device=device, dtype=dtype)
            return t

        if t_type == "log":
            # Logistic-normal -> (0,1)
            normal_samples = torch.randn(batch_size, 1, device=device, dtype=dtype) * std + (-mean)
            t = 1 / (1 + torch.exp(-normal_samples))
            return t

        if t_type == "default":
            # Ported shape logic from FACM, simplified for (B,1)
            sigma = torch.randn(batch_size, 1, device=device, dtype=dtype)
            sigma = (sigma * std + (-mean)).exp()
            t = torch.arctan(sigma) * (2.0 / math.pi)
            return t

        raise ValueError(f"Invalid t_type: {t_type}")

    # -------------------------
    # Loss functions (FACM)
    # -------------------------
    @staticmethod
    def _norm_l2_loss(pred: torch.Tensor, target: torch.Tensor, p: float = 0.5, c: float = 1e-3) -> torch.Tensor:
        e = _mean_flat((pred - target) ** 2)
        return e / (e + c).pow(p).detach()

    @staticmethod
    def _flow_matching_loss(pred: torch.Tensor, target: torch.Tensor, use_cosine: bool = True) -> torch.Tensor:
        mse = _mean_flat((pred - target) ** 2)
        if not use_cosine:
            return mse
        cos = 1 - F.cosine_similarity(pred, target, dim=1, eps=1e-8)
        return mse + cos

    def loss_function(
        self,
        x: torch.Tensor,
        *,
        config: Optional[FACMConfig] = None,
        t: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute FACM loss (total, cm, fm) for a batch of data x.

        Args:
            x: data samples, (B, D)
            config: FACMConfig
            t: optional time tensor, (B,1). If None, sampled according to config.
            z: optional noise tensor, (B, D). If None, sampled from N(0, I). For CFM coupling.
            weights: optional loss weights for UOTRFM (B,)
        """
        if config is None:
            config = FACMConfig()

        b = x.shape[0]
        device = x.device

        # Noise and time
        if z is None:
            z = torch.randn_like(x)
        if t is None:
            t = self.sample_t(b, device, t_type=config.t_type, mean=config.mean, std=config.std, dtype=x.dtype)
        t = t.view(b, 1)

        # Rectified flow linear interpolant: x_t = (1-t)*z + t*x
        x_t = (1 - t) * z + t * x
        v = x - z  # ground-truth flow velocity

        # FM anchor: use r=t (as in FACM code: compiled_model(x_t, t_fm, t_fm))
        r_fm = t
        F_fm = self.forward(x_t, t, r_fm)
        fm_loss = self._flow_matching_loss(F_fm, v, use_cosine=config.use_cosine_in_fm)

        # CM accelerator: use r=1 (end time) and compute JVP along (v, 1)
        r_cm = torch.ones_like(t)

        def model_wrapper(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return self.forward(x_in, t_in, r_cm)

        v_x = v
        v_t = torch.ones_like(t)
        F_avg, F_avg_grad = torch.func.jvp(model_wrapper, (x_t, t), (v_x, v_t))

        F_avg_grad = F_avg_grad.detach()
        F_avg_sg = F_avg.detach()

        # v_bar = v + (t_end - t) * d/ds F(x_t + s*v, t + s)
        v_bar = v + (1 - t) * F_avg_grad
        g = F_avg_sg - v_bar

        alpha = 1 - t.pow(config.p)
        if config.clamp_g is not None and config.clamp_g > 0:
            g = g.clamp(min=-config.clamp_g, max=config.clamp_g)
        target = F_avg_sg - alpha * g

        beta = torch.cos(t.flatten() * math.pi / 2.0)
        cm_loss = self._norm_l2_loss(F_avg, target, p=config.p, c=config.c) * beta

        # Apply weights for UOTRFM
        if weights is not None:
            weights_expanded = weights.view(-1, 1) if weights.dim() == 1 else weights
            # Weight only the FM loss (as per UOTRFM)
            fm_loss = fm_loss * weights_expanded.squeeze()
            cm_loss = cm_loss * weights_expanded.squeeze()

        total = cm_loss.mean() + fm_loss.mean()
        return total, cm_loss.mean(), fm_loss.mean()

    # -------------------------
    # Samplers (adapted from FACM sampler.py)
    # -------------------------
    @staticmethod
    def _apply_timestep_shift(t_steps: torch.Tensor, timestep_shift: float) -> torch.Tensor:
        if timestep_shift <= 0:
            return t_steps
        # tm = (s * t) / (1 + (s-1) * t)
        s = timestep_shift
        return (s * t_steps) / (1 + (s - 1) * t_steps)

    @torch.no_grad()
    def sample_euler(
        self,
        z0: torch.Tensor,
        *,
        n_steps: int = 20,
        heun: bool = False,
        timestep_shift: float = 0.0,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Euler (optionally Heun) sampler for rectified flow: integrate from t=0 to t=1.
        Returns trajectory for visualization: (n_steps+1, B, D).
        """
        self.eval()
        if device is None:
            device = str(z0.device)

        b = z0.shape[0]
        x_next = z0.to(torch.float64)

        t_steps = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64, device=x_next.device)
        t_steps = self._apply_timestep_shift(t_steps, timestep_shift)

        traj = torch.zeros(n_steps + 1, b, self.input_dim, device=x_next.device, dtype=torch.float64)
        traj[0] = x_next

        for i, (t_cur, t_nxt) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            t_in = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * t_cur
            r_in = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * t_cur

            d_cur = self.forward(
                x_cur.to(dtype=z0.dtype),
                t_in.to(dtype=z0.dtype),
                r_in.to(dtype=z0.dtype),
            ).to(torch.float64)

            x_next = x_cur + (t_nxt - t_cur) * d_cur

            if heun and i < n_steps - 1:
                t_in2 = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * t_nxt
                r_in2 = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * t_nxt
                d_prime = self.forward(
                    x_next.to(dtype=z0.dtype),
                    t_in2.to(dtype=z0.dtype),
                    r_in2.to(dtype=z0.dtype),
                ).to(torch.float64)
                x_next = x_cur + (t_nxt - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

            traj[i + 1] = x_next

        return traj.to(dtype=z0.dtype)

    @torch.no_grad()
    def sample_consistency(
        self,
        z0: torch.Tensor,
        *,
        n_steps: int = 1,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Consistency-model sampler (FACM sampler.py style).
        Returns final x_end (B, D).
        """
        self.eval()
        if device is None:
            device = str(z0.device)

        b = z0.shape[0]
        x_next = z0.to(torch.float64)
        t_steps = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64, device=x_next.device)

        x_end = None
        for (t_cur, t_nxt) in zip(t_steps[:-1], t_steps[1:]):
            x_cur = x_next
            t_in = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * t_cur
            r_in = torch.ones(b, 1, device=x_cur.device, dtype=torch.float64) * 1.0

            d_cur = self.forward(
                x_cur.to(dtype=z0.dtype),
                t_in.to(dtype=z0.dtype),
                r_in.to(dtype=z0.dtype),
            ).to(torch.float64)

            x_end = x_cur + (t_steps[-1] - t_cur) * d_cur
            noise = torch.randn_like(x_end)
            x_next = t_nxt * x_end + (1 - t_nxt) * noise

        assert x_end is not None
        return x_end.to(dtype=z0.dtype)

