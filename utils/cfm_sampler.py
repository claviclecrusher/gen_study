"""
Conditional Flow Matching (CFM) Sampler
Implements various coupling strategies for flow-based generative models.

Supported methods:
- icfm: Independent CFM (random coupling, default)
- otcfm: Optimal Transport CFM (exact OT coupling)
- uotcfm: Unbalanced OT CFM (unbalanced Sinkhorn coupling)
- uotrfm: Unbalanced OT Reweighted FM (uotcfm + loss weighting)

Reference: torchcfm library
"""
import numpy as np
import torch
import ot as pot
from functools import partial
from typing import Optional, Tuple, Union


class CFMSampler:
    """
    CFM Sampler for computing couplings between source and target distributions.
    
    Supports multiple coupling strategies:
    - icfm: Independent random coupling (identity permutation)
    - otcfm: Exact optimal transport coupling
    - uotcfm: Unbalanced optimal transport coupling
    - uotrfm: Unbalanced OT with reweighted loss
    
    Args:
        method: Coupling method ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
        reg: Entropic regularization for Sinkhorn (default: 0.05)
        reg_m: Marginal regularization for unbalanced OT (default: (inf, 2.0))
        normalize_cost: Whether to normalize cost matrix (default: True)
        weight_power: Power factor for UOTRFM weights (default: 10)
    """
    
    def __init__(
        self,
        method: str = 'icfm',
        reg: float = 0.05,
        reg_m: Tuple[float, float] = (float('inf'), 2.0),
        normalize_cost: bool = True,
        weight_power: float = 10.0
    ):
        self.method = method.lower()
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.weight_power = weight_power
        
        # Validate method
        valid_methods = ['icfm', 'otcfm', 'uotcfm', 'uotrfm']
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method: {method}. Valid methods: {valid_methods}")
        
        # Setup OT solver based on method
        if self.method == 'otcfm':
            self.ot_fn = partial(pot.emd, numThreads=1)
        elif self.method in ['uotcfm', 'uotrfm']:
            self.ot_fn = partial(
                pot.unbalanced.sinkhorn_knopp_unbalanced,
                reg=reg,
                reg_m=reg_m,
                log=True
            )
    
    def _compute_cost_matrix(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """Compute squared Euclidean cost matrix between x0 and x1."""
        # Flatten if needed
        if x0.dim() > 2:
            x0_flat = x0.reshape(x0.shape[0], -1)
        else:
            x0_flat = x0
        if x1.dim() > 2:
            x1_flat = x1.reshape(x1.shape[0], -1)
        else:
            x1_flat = x1
        
        # Compute squared Euclidean distance
        M = torch.cdist(x0_flat, x1_flat) ** 2
        
        # Normalize if needed
        if self.normalize_cost:
            M = M / (M.max() + 1e-12)
        
        return M.detach().cpu().numpy()
    
    def _get_ot_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[np.ndarray, Optional[dict]]:
        """Compute OT plan between x0 and x1."""
        a = pot.unif(x0.shape[0])
        b = pot.unif(x1.shape[0])
        M = self._compute_cost_matrix(x0, x1)
        
        result = self.ot_fn(a, b, M)
        
        # Handle different return types
        if isinstance(result, tuple) and len(result) == 2:
            pi, log_uv = result
        else:
            pi = result
            log_uv = None
        
        # Check for numerical errors
        if not np.all(np.isfinite(pi)):
            raise RuntimeError("Numerical error in OT plan: non-finite values")
        if np.abs(pi.sum()) < 1e-8:
            raise RuntimeError("Numerical error in OT plan: zero sum")
        
        return pi, log_uv
    
    def _sample_from_plan(self, pi: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample indices from OT plan."""
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1],
            p=p,
            size=batch_size,
            replace=False
        )
        return np.divmod(choices, pi.shape[1])
    
    def sample_coupling(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample coupled pairs (x0, x1) according to the specified method.
        
        Args:
            x0: Source samples (batch_size, *dim)
            x1: Target samples (batch_size, *dim)
        
        Returns:
            x0_coupled: Coupled source samples
            x1_coupled: Coupled target samples
            weights: Loss weights (only for uotrfm, None otherwise)
        """
        batch_size = x0.shape[0]
        
        if self.method == 'icfm':
            # Independent coupling: identity permutation (no change)
            return x0, x1, None
        
        elif self.method == 'otcfm':
            # Exact OT coupling
            pi, _ = self._get_ot_plan(x0, x1)
            i, j = self._sample_from_plan(pi, batch_size)
            return x0[i], x1[j], None
        
        elif self.method == 'uotcfm':
            # Unbalanced OT coupling (no weights)
            pi, _ = self._get_ot_plan(x0, x1)
            i, j = self._sample_from_plan(pi, batch_size)
            return x0[i], x1[j], None
        
        elif self.method == 'uotrfm':
            # Unbalanced OT with reweighted loss
            pi, _ = self._get_ot_plan(x0, x1)
            i, j = self._sample_from_plan(pi, batch_size)
            
            # Compute weights: inverse of target marginal (tnu)
            # tnu = pi.sum(dim=0) - marginal distribution over target
            pi_tensor = torch.tensor(pi, dtype=x0.dtype, device=x0.device)
            tnu = pi_tensor.sum(dim=0)  # (batch_size,)
            
            # Normalize by batch size
            tnu = tnu / (1.0 / batch_size)
            
            # Inverse weight (minority samples get higher weight)
            fm_weight = 1.0 / (tnu + 1e-8)
            
            # Select weights for recoupled target samples
            fm_weight = fm_weight[j]
            
            # Apply power factor
            fm_weight = fm_weight ** self.weight_power
            
            # Reshape for broadcasting with loss
            # Keep as 1D tensor, will be reshaped in training loop as needed
            weights = fm_weight.to(x0.device)
            
            return x0[i], x1[j], weights
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def __repr__(self):
        return (f"CFMSampler(method='{self.method}', reg={self.reg}, "
                f"reg_m={self.reg_m}, weight_power={self.weight_power})")


def create_cfm_sampler(
    cfm_type: str = 'icfm',
    reg: float = 0.05,
    reg_m: Tuple[float, float] = (float('inf'), 2.0),
    weight_power: float = 10.0
) -> CFMSampler:
    """
    Factory function to create CFM sampler.
    
    Args:
        cfm_type: Type of CFM ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
        reg: Entropic regularization for Sinkhorn
        reg_m: Marginal regularization for unbalanced OT
        weight_power: Power factor for UOTRFM weights
    
    Returns:
        CFMSampler instance
    """
    return CFMSampler(
        method=cfm_type,
        reg=reg,
        reg_m=reg_m,
        weight_power=weight_power
    )


if __name__ == "__main__":
    # Test CFM sampler
    print("Testing CFM Sampler...")
    
    # Create test data
    torch.manual_seed(42)
    x0 = torch.randn(64, 2)  # Source (noise)
    x1 = torch.randn(64, 2) + 2.0  # Target (data)
    
    for method in ['icfm', 'otcfm', 'uotcfm', 'uotrfm']:
        print(f"\n--- Testing {method.upper()} ---")
        sampler = create_cfm_sampler(cfm_type=method)
        print(sampler)
        
        x0_c, x1_c, weights = sampler.sample_coupling(x0, x1)
        print(f"x0_coupled shape: {x0_c.shape}")
        print(f"x1_coupled shape: {x1_c.shape}")
        
        if weights is not None:
            print(f"weights shape: {weights.shape}")
            print(f"weights mean: {weights.mean():.4f}, std: {weights.std():.4f}")
        else:
            print("weights: None")
    
    print("\n✓ All tests passed!")
