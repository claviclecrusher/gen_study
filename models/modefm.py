"""
ModeFlowMatching (ModeFM) model - Flow Matching with Gaussian Kernel Loss

Inherits FlowMatching architecture but uses Gaussian Kernel Loss (Correntropy) instead of L2/MSE.
Sigma annealing is configurable for mode-seeking behavior.

Does NOT support UOTRFM coupling - raises exception if cfm_type='uotrfm'.
"""
import torch
import torch.nn as nn
from models.flow_matching import FlowMatching


class ModeFlowMatching(FlowMatching):
    """
    Flow Matching with Gaussian Kernel Loss (Mode-seeking)

    Same architecture as FlowMatching but uses Correntropy-based loss:
    L = mean(1 - exp(-||v_pred - u_t||^2 / (2*sigma^2)))

    Sigma annealing: smaller sigma -> more mode-seeking, larger sigma -> more mean-seeking.
    """

    def __init__(self, input_dim=1, hidden_dims=None, initial_sigma=5.0):
        super(ModeFlowMatching, self).__init__(input_dim=input_dim, hidden_dims=hidden_dims)
        self.sigma = initial_sigma

    def update_sigma(self, new_sigma):
        """Update sigma for annealing (called each epoch during training)."""
        self.sigma = new_sigma

    def loss_function(self, z, x, t, weights=None, sigma_adaptive_params=None):
        """
        Compute Gaussian Kernel Loss (Correntropy): mean(1 - exp(-e_sq / (2*sigma^2)))

        UOTRFM is NOT supported - raises ValueError if weights is not None.
        (ModeFM uses icfm, otcfm, or uotcfm where weights are typically None)

        Args:
            z: Source samples from N(0, I) (batch_size, input_dim)
            x: Target data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            weights: Must be None (UOTRFM not supported)
            sigma_adaptive_params: If not None, use batch-statistics adaptive bandwidth.
                Dict with keys: gamma (scaling), q (quantile, default 0.5), eps (min sigma).

        Returns:
            loss: Gaussian kernel loss (tensor)
            sigma_used: Float sigma used (for logging). When adaptive, computed per batch.
        """
        if weights is not None:
            raise ValueError(
                "ModeFM does not support UOTRFM coupling. "
                "Use icfm, otcfm, or uotcfm. Please set cfm_type to one of these."
            )

        # Compute conditional flow
        x_t, u_t = self.compute_conditional_flow(z, x, t)

        # Predict velocity
        v_pred = self.forward(x_t, t)

        # Error vector and squared Euclidean distance
        error_vector = v_pred - u_t
        e_sq = torch.sum(error_vector ** 2, dim=1)  # (batch_size,) squared L2 norm

        if sigma_adaptive_params is not None:
            # Batch-Statistics Adaptive Bandwidth
            # Sigma from batch residual distribution, detached to prevent trivial solution (sigma->inf)
            gamma = sigma_adaptive_params.get('gamma', 1.0)
            q = sigma_adaptive_params.get('q', 0.5)
            eps = sigma_adaptive_params.get('eps', 1e-6)

            # residuals = L2 norm per sample (sqrt of e_sq)
            residuals = torch.sqrt(e_sq + 1e-12)

            with torch.no_grad():
                sigma_base = torch.quantile(residuals.float(), q=q)
                sigma = gamma * sigma_base
                sigma = max(float(sigma), eps)

            # Ensure sigma is detached and not part of gradient graph
            sigma_tensor = torch.tensor(sigma, dtype=v_pred.dtype, device=v_pred.device)
            sigma_used = sigma
        else:
            # Fixed/annealed sigma
            sigma = max(1e-6, self.sigma)
            sigma_tensor = sigma
            sigma_used = sigma

        # Gaussian kernel: exp(-e_sq / (2 * sigma^2))
        kernel_val = torch.exp(-e_sq / (2 * sigma_tensor ** 2))

        # Correntropy maximization = (1 - Correntropy) minimization
        loss = torch.mean(1.0 - kernel_val)

        return loss, sigma_used
