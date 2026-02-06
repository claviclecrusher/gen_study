"""
TopK-OTCFM (TopK Optimal Transport Conditional Flow Matching) model

Key idea:
- Pretraining: Standard OTCFM training
- Retraining: Use OT coupling + ODE solver to estimate x1' from x0, then compute ||x1' - x1||²
- Only update top k samples with smallest errors (most accurate predictions)
- Goal: Learn paths that are closer to straight lines

Reference: TopK-CFM approach
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Please install with: pip install torchdiffeq")


class TopKFlowMatching(nn.Module):
    """
    TopK-OTCFM model
    
    Training stages:
    1. Pretraining: Standard OTCFM loss (Flow Matching with OT coupling)
    2. Retraining: OT coupling + ODE simulation + TopK selection
    
    Args:
        input_dim: Dimension of input space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
        ode_solver: ODE solver method ('dopri5' or 'euler', default: 'dopri5')
        ode_tol: Tolerance for adaptive ODE solver (default: 1e-5)
    """
    
    def __init__(self, input_dim=1, hidden_dims=None, ode_solver='dopri5', ode_tol=1e-5):
        super(TopKFlowMatching, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 32]
        
        self.input_dim = input_dim
        self.ode_solver = ode_solver
        self.ode_tol = ode_tol
        
        if not TORCHDIFFEQ_AVAILABLE and ode_solver == 'dopri5':
            print("Warning: torchdiffeq not available, falling back to Euler solver")
            self.ode_solver = 'euler'
        
        # Velocity network v_θ(x_t, t)
        # Input: [x_t, t] concatenated
        self.velocity_net = BaseMLP(
            input_dim=input_dim + 1,  # x_t + t
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )
    
    def forward(self, x_t, t):
        """
        Compute velocity field v_θ(x_t, t)
        
        Args:
            x_t: Current position (batch_size, input_dim)
            t: Time (batch_size, 1) or scalar tensor
            
        Returns:
            v: Velocity field (batch_size, input_dim)
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x_t.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Concatenate x_t and t
        input_vec = torch.cat([x_t, t], dim=-1)
        v = self.velocity_net(input_vec)
        return v
    
    def compute_conditional_flow(self, z, x, t):
        """
        Compute conditional flow x_t = (1-t) * z + t * x
        
        Args:
            z: Source samples from N(0, I) (batch_size, input_dim)
            x: Target data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            
        Returns:
            x_t: Interpolated samples (batch_size, input_dim)
            u_t: Conditional velocity u_t = x - z (batch_size, input_dim)
        """
        x_t = (1 - t) * z + t * x
        u_t = x - z  # Conditional velocity (constant for linear interpolation)
        return x_t, u_t
    
    def loss_function_pretrain(self, z, x, t, weights=None):
        """
        Compute standard Flow Matching loss for pretraining: ||v_θ(x_t, t) - u_t||^2
        
        Args:
            z: Source samples from N(0, I) (batch_size, input_dim)
            x: Target data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            weights: Optional loss weights for UOTRFM (batch_size,)
            
        Returns:
            loss: Flow matching loss
        """
        # Compute conditional flow
        x_t, u_t = self.compute_conditional_flow(z, x, t)
        
        # Predict velocity
        v_pred = self.forward(x_t, t)
        
        # MSE loss between predicted and true velocity
        if weights is not None:
            weights = weights.view(-1, 1)  # (batch_size, 1)
            loss = (weights * (v_pred - u_t) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(v_pred, u_t, reduction='mean')
        
        return loss
    
    def solve_ode(self, x0, t_span, device='cpu', return_nfe=False):
        """
        Solve ODE: dx/dt = v_θ(x_t, t) from t=0 to t=1
        
        Args:
            x0: Initial condition (batch_size, input_dim)
            t_span: Time span, e.g., torch.tensor([0.0, 1.0])
            device: Device to run on
            return_nfe: If True, return NFE along with solution
            
        Returns:
            x1: Final state at t=1 (batch_size, input_dim)
            nfe: Number of function evaluations (if return_nfe=True)
        """
        self.eval()
        
        # Counter for function evaluations
        nfe_counter = {'count': 0}
        
        def ode_func(t, x):
            # x: (batch_size, input_dim)
            # t: scalar tensor
            # Return: (batch_size, input_dim)
            nfe_counter['count'] += 1
            t_batch = torch.ones(x.shape[0], 1, device=device) * t
            return self.forward(x, t_batch)
        
        if self.ode_solver == 'dopri5' and TORCHDIFFEQ_AVAILABLE:
            # Use adaptive dopri5 solver
            with torch.no_grad():
                solution = odeint(
                    ode_func,
                    x0,
                    t_span,
                    method='dopri5',
                    rtol=self.ode_tol,
                    atol=self.ode_tol
                )
            x1 = solution[-1]  # Return final state
            nfe = nfe_counter['count']
        else:
            # Use Euler solver (fallback)
            n_steps = 100
            dt = (t_span[1] - t_span[0]) / n_steps
            x = x0.clone()
            
            with torch.no_grad():
                for step in range(n_steps):
                    t = t_span[0] + step * dt
                    t_batch = torch.ones(x.shape[0], 1, device=device) * t
                    v = self.forward(x, t_batch)
                    x = x + dt * v
            
            x1 = x
            nfe = n_steps  # For Euler, NFE = number of steps
        
        if return_nfe:
            return x1, nfe
        else:
            return x1
    
    def loss_function_retrain(self, x0, x1, t, top_filter_k=1.0, device='cpu', weights=None):
        """
        Compute retraining loss with TopK selection
        
        Algorithm:
        1. OT coupling gives (x0, x1) pairs
        2. Solve ODE from x0 to get x1' (estimated target) - NO GRADIENT
        3. Compute error = ||x1' - x1||² for each sample (selection criterion only)
        4. Select top k samples with smallest errors (best predicted)
        5. Compute FM loss ONLY for selected samples
        
        Key insight: ODE error is only for SELECTION, actual loss is FM loss!
        
        Args:
            x0: Source samples (batch_size, input_dim)
            x1: Target samples (coupled with x0 via OT) (batch_size, input_dim)
            t: Time for FM loss (batch_size, 1)
            top_filter_k: Fraction of samples to update (0 < k <= 1, default: 1.0)
            device: Device to run on
            weights: Optional loss weights for UOTRFM (batch_size,)
            
        Returns:
            loss: FM loss for TopK selected samples
            logs: Dictionary with loss components
        """
        batch_size = x0.shape[0]
        
        # Step 1: Solve ODE from x0 to estimate x1' (NO GRADIENT - only for selection)
        with torch.no_grad():
            t_span = torch.tensor([0.0, 1.0], device=device)
            x1_pred, nfe = self.solve_ode(x0, t_span, device=device, return_nfe=True)
            
            # Step 2: Compute L2 distance for each sample (selection criterion only)
            # error[i] = ||x1_pred[i] - x1[i]||²
            error = torch.sum((x1_pred - x1) ** 2, dim=1)  # (batch_size,)
            
            # Step 3: Select top k samples with smallest errors
            k = max(1, int(batch_size * top_filter_k))
            k = min(k, batch_size)  # Ensure k <= batch_size
            
            # Get indices of top k smallest errors (best predicted samples)
            _, topk_indices = torch.topk(error, k, largest=False)  # smallest k errors
        
        # Step 4: Select top k samples for training
        x0_selected = x0[topk_indices]
        x1_selected = x1[topk_indices]
        t_selected = t[topk_indices]
        
        # Step 5: Compute FM loss for selected samples (WITH GRADIENT)
        x_t, u_t = self.compute_conditional_flow(x0_selected, x1_selected, t_selected)
        v_pred = self.forward(x_t, t_selected)
        
        if weights is not None:
            weights_selected = weights[topk_indices].view(-1, 1)
            loss = (weights_selected * (v_pred - u_t) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(v_pred, u_t, reduction='mean')
        
        # Log statistics
        return loss, {
            'loss_fm': loss.item(),
            'k_selected': k,
            'k_fraction': top_filter_k,
            'mean_ode_error': error.mean().item(),
            'min_ode_error': error.min().item(),
            'max_ode_error': error.max().item(),
            'topk_ode_error': error[topk_indices].mean().item(),
            'nfe': nfe  # Number of function evaluations for ODE solver
        }
    
    def sample(self, z, n_steps=100, device='cpu'):
        """
        Sample from the model using ODE solver
        
        Args:
            z: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (for Euler, ignored for dopri5)
            device: Device to run on
            
        Returns:
            trajectory: Full trajectory from t=0 to t=1 (n_steps+1, n_samples, input_dim)
        """
        self.eval()
        
        n_samples = z.shape[0]
        
        if self.ode_solver == 'dopri5' and TORCHDIFFEQ_AVAILABLE:
            # Use adaptive dopri5 solver
            t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
            
            def ode_func(t, x):
                t_batch = torch.ones(x.shape[0], 1, device=device) * t
                return self.forward(x, t_batch)
            
            with torch.no_grad():
                trajectory = odeint(
                    ode_func,
                    z,
                    t_span,
                    method='dopri5',
                    rtol=self.ode_tol,
                    atol=self.ode_tol
                )
            # trajectory shape: (n_steps+1, n_samples, input_dim)
            return trajectory
        else:
            # Use Euler solver
            dt = 1.0 / n_steps
            
            # Store trajectory
            trajectory = torch.zeros(n_steps + 1, n_samples, self.input_dim).to(device)
            trajectory[0] = z
            
            x_t = z.clone()
            
            with torch.no_grad():
                for step in range(n_steps):
                    t = torch.ones(n_samples, 1).to(device) * (step * dt)
                    
                    # Euler step: x_{t+1} = x_t + dt * v_θ(x_t, t)
                    v = self.forward(x_t, t)
                    x_t = x_t + dt * v
                    
                    trajectory[step + 1] = x_t
            
            return trajectory


if __name__ == "__main__":
    # Test TopK-OTCFM
    print("Testing TopK-OTCFM...")
    
    # Create model
    model = TopKFlowMatching(input_dim=1, ode_solver='dopri5')
    print(f"Model architecture:\n{model}")
    
    # Test forward pass
    batch_size = 10
    x_t = torch.randn(batch_size, 1)
    t = torch.rand(batch_size, 1)
    v = model(x_t, t)
    print(f"\nVelocity v shape: {v.shape}")
    
    # Test pretraining loss
    z = torch.randn(batch_size, 1)
    x = torch.randn(batch_size, 1)
    loss_pretrain = model.loss_function_pretrain(z, x, t)
    print(f"\nPretraining loss: {loss_pretrain.item():.4f}")
    
    # Test retraining loss
    x0 = torch.randn(batch_size, 1)
    x1 = torch.randn(batch_size, 1)
    t_retrain = torch.rand(batch_size, 1)
    loss_retrain, logs = model.loss_function_retrain(x0, x1, t_retrain, top_filter_k=0.5, device='cpu')
    print(f"\nRetraining FM loss (topk=0.5): {loss_retrain.item():.4f}")
    print(f"  Mean ODE error: {logs['mean_ode_error']:.4f}")
    print(f"  TopK ODE error: {logs['topk_ode_error']:.4f}")
    print(f"  Selected k: {logs['k_selected']}")
    
    # Test sampling
    z_test = torch.randn(5, 1)
    trajectory = model.sample(z_test, n_steps=100, device='cpu')
    print(f"\nTrajectory shape: {trajectory.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params}")
