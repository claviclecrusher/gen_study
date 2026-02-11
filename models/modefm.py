"""
ModeFlowMatching (ModeFM) model - Flow Matching with Gaussian Kernel Loss

Inherits FlowMatching architecture but uses Gaussian Kernel Loss (Correntropy) instead of L2/MSE.
Sigma annealing is configurable for mode-seeking behavior.

Optional var_head: predicts per-sample variance for sample-adaptive sigma.

Does NOT support UOTRFM coupling - raises exception if cfm_type='uotrfm'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.flow_matching import FlowMatching


class ModeFlowMatching(FlowMatching):
    """
    Flow Matching with Gaussian Kernel Loss (Mode-seeking)

    Same architecture as FlowMatching but uses Correntropy-based loss:
    L = mean(1 - exp(-||v_pred - u_t||^2 / (2*sigma^2)))

    Optional var_head: when use_var_head=True, refactors into shared backbone + v_head + var_head.
    var_head predicts conditional velocity variance for sample-adaptive sigma.
    """

    def __init__(self, input_dim=1, hidden_dims=None, initial_sigma=5.0, use_var_head=False):
        super(ModeFlowMatching, self).__init__(input_dim=input_dim, hidden_dims=hidden_dims)
        self.sigma = initial_sigma
        self.use_var_head = use_var_head

        if use_var_head:
            # Refactor: backbone + v_head + var_head (shared backbone, two heads)
            if hidden_dims is None:
                hidden_dims = [32, 64, 32]
            hidden_dim = hidden_dims[-1]

            # Build backbone (all hidden layers from original velocity_net, no output layer)
            backbone_layers = []
            prev_dim = input_dim + 1
            for h in hidden_dims:
                backbone_layers.append(nn.Linear(prev_dim, h))
                backbone_layers.append(nn.ReLU())
                prev_dim = h
            self.backbone = nn.Sequential(*backbone_layers)

            self.v_head = nn.Linear(hidden_dim, input_dim)
            self.var_head = nn.Linear(hidden_dim, 1)
            # Remove original velocity_net (replaced by backbone + heads)
            del self.velocity_net

    def forward(self, x_t, t):
        """
        Compute velocity field v_θ(x_t, t).
        When use_var_head, also return variance prediction var_θ.
        """
        input_vec = torch.cat([x_t, t], dim=-1)

        if self.use_var_head:
            h = self.backbone(input_vec)
            v_pred = self.v_head(h)
            raw_var = self.var_head(h)
            var_pred = F.softplus(raw_var) + 1e-6  # (batch_size, 1), always positive
            return v_pred, var_pred, h
        else:
            return self.velocity_net(input_vec)

    def sample(self, z, n_steps=100, device='cpu'):
        """Override: handle forward returning (v, var) when use_var_head."""
        self.eval()
        n_samples = z.shape[0]
        dt = 1.0 / n_steps
        trajectory = torch.zeros(n_steps + 1, n_samples, self.input_dim).to(device)
        trajectory[0] = z
        x_t = z.clone()

        with torch.no_grad():
            for step in range(n_steps):
                t = torch.ones(n_samples, 1).to(device) * (step * dt)
                out = self.forward(x_t, t)
                v = out[0] if isinstance(out, tuple) else out
                x_t = x_t + dt * v
                trajectory[step + 1] = x_t

        return trajectory

    def update_sigma(self, new_sigma):
        """Update sigma for annealing (called each epoch during training)."""
        self.sigma = new_sigma

    def loss_function(self, z, x, t, weights=None, sigma_adaptive_params=None,
                      sample_adaptive_params=None, var_loss_weight=1.0):
        """
        Compute Gaussian Kernel Loss (Correntropy): mean(1 - exp(-e_sq / (2*sigma^2)))

        UOTRFM is NOT supported - raises ValueError if weights is not None.

        Args:
            z: Source samples from N(0, I) (batch_size, input_dim)
            x: Target data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            weights: Must be None (UOTRFM not supported)
            sigma_adaptive_params: If not None, use batch-statistics adaptive bandwidth.
            sample_adaptive_params: If not None and use_var_head, use var_head output for sigma.
                Dict: scale (kernel_width = scale * 2 * sigma_sq), eps.
            var_loss_weight: Weight for variance head loss when use_var_head (default 1.0).

        Returns:
            loss: Total loss (MCC + var_loss_weight * loss_var when var_head)
            sigma_used: Float sigma used for logging (batch mean when sample_adaptive).
        """
        if weights is not None:
            raise ValueError(
                "ModeFM does not support UOTRFM coupling. "
                "Use icfm, otcfm, or uotcfm. Please set cfm_type to one of these."
            )

        # Compute conditional flow
        x_t, u_t = self.compute_conditional_flow(z, x, t)

        # Predict velocity (and var when use_var_head)
        out = self.forward(x_t, t)
        if self.use_var_head:
            v_pred, var_pred, h = out  # var_pred: (batch_size, 1), h: backbone output
        else:
            v_pred = out

        # Error vector and squared Euclidean distance
        error_vector = v_pred - u_t
        e_sq = torch.sum(error_vector ** 2, dim=1)  # (batch_size,)
        squared_error = e_sq.unsqueeze(1)  # (batch_size, 1) for loss_var

        # Variance head loss (when use_var_head): || var_theta - stop_grad(||v-u||^2) ||^2
        # Gradient flows only through var_head, not through shared backbone (use h.detach())
        loss_var = None
        if self.use_var_head:
            var_pred_for_var = F.softplus(self.var_head(h.detach())) + 1e-6
            loss_var = F.mse_loss(var_pred_for_var, squared_error.detach())

        # MCC sigma and kernel
        if sample_adaptive_params is not None and self.use_var_head:
            scale = sample_adaptive_params.get('scale', 1.0)
            eps = sample_adaptive_params.get('eps', 1e-8)
            warmup_epochs = sample_adaptive_params.get('warmup_epochs', 0)
            warmup_sigma = sample_adaptive_params.get('warmup_sigma', 10.0)
            current_epoch = sample_adaptive_params.get('epoch', 0)

            # Warmup: use fixed sigma for first warmup_epochs, then switch to var_head output
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                sigma = warmup_sigma
                kernel_width = scale * 2.0 * (sigma ** 2) + eps
                loss_mcc_elements = 1.0 - torch.exp(-e_sq.unsqueeze(1) / kernel_width)
                loss_mcc = loss_mcc_elements.mean()
                sigma_used = sigma
            else:
                # Sample-adaptive: use var_head output. sigma^2 = var_pred
                sigma_sq = var_pred.detach()  # (batch_size, 1), stop gradient
                kernel_width = scale * 2.0 * sigma_sq + eps  # (batch_size, 1)
                loss_mcc_elements = 1.0 - torch.exp(-e_sq.unsqueeze(1) / kernel_width)
                loss_mcc = loss_mcc_elements.mean()
                sigma_used = float(torch.sqrt(sigma_sq.mean()).item())
        elif sigma_adaptive_params is not None:
            # Batch-Statistics Adaptive Bandwidth
            gamma = sigma_adaptive_params.get('gamma', 1.0)
            q = sigma_adaptive_params.get('q', 0.5)
            eps = sigma_adaptive_params.get('eps', 1e-6)
            residuals = torch.sqrt(e_sq + 1e-12)
            with torch.no_grad():
                sigma_base = torch.quantile(residuals.float(), q=q)
                sigma = gamma * sigma_base
                sigma = max(float(sigma), eps)
            sigma_tensor = torch.tensor(sigma, dtype=v_pred.dtype, device=v_pred.device)
            kernel_val = torch.exp(-e_sq / (2 * sigma_tensor ** 2))
            loss_mcc = torch.mean(1.0 - kernel_val)
            sigma_used = sigma
        else:
            # Fixed/annealed sigma
            sigma = max(1e-6, self.sigma)
            sigma_tensor = sigma
            kernel_val = torch.exp(-e_sq / (2 * sigma_tensor ** 2))
            loss_mcc = torch.mean(1.0 - kernel_val)
            sigma_used = sigma

        # Total loss (var_loss_weight=0 means do not add loss_var, e.g. when caller does not want it)
        if loss_var is not None and var_loss_weight > 0:
            loss = loss_mcc + var_loss_weight * loss_var
        else:
            loss = loss_mcc

        return loss, sigma_used
