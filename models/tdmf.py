"""
Translation Decoupled MeanFlow (TDMF) model
Based on Improved MeanFlow (iMF) with Translation Centered Loss (L_TC)

Key idea:
- MeanFlow tends to learn easy translation instead of proper velocity integration
- Gradient Spatial Centering: modify gradients to remove translation component
- Implements L_TC (Translation Centered Loss) via backward hook

Mathematical formulation:
    L_FM = (1/B) Σ_{i=1}^B ||V_i - v_target_i||^2  (standard Flow Matching loss)
    
    G_i = ∇_{u_θ} L_FM  (gradient for sample i)
    Ḡ = (1/B) Σ_{i=1}^B G_i  (batch mean gradient)
    G̃_i = G_i - (1 - λ_trans) · Ḡ  (centered gradient)
    
    Optimization uses G̃ instead of G, implementing L_TC

Reference: iMF base - https://arxiv.org/html/2512.02012v1
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class TranslationDecoupledLoss(nn.Module):
    """
    Translation Centered Loss (L_TC) for 1D/2D point data
    
    Implements Gradient Spatial Centering via loss decomposition.
    
    Mathematical formulation:
        Standard Flow Matching: L_FM = (1/B) Σ_{i=1}^B ||pred_i - target_i||^2
        
        Translation Centered Loss: 
        L_TC = (1/B) Σ_{i=1}^B ||pred_i - target_i - (pred̄ - target̄)||^2 
               + λ_trans · ||pred̄ - target̄||^2
        
        where pred̄ = (1/B) Σ pred_i, target̄ = (1/B) Σ target_i
        
        This loss decomposition achieves gradient centering:
        ∇ L_TC = ∇ L_FM - (1 - λ_trans) · Ḡ
        where Ḡ = batch mean gradient
        
        Equivalent to: G̃ = G - (1 - λ_trans) · Ḡ (gradient centering)
        
    Args:
        lambda_trans (float): Weight for translation gradient suppression.
            - 0.0: Complete gradient centering (G̃ = G - Ḡ), removes all translation
            - 1.0: No centering (G̃ = G), standard Flow Matching
            - 0.1 (default): Partial centering, suppresses 90% of translation gradient
    """
    
    def __init__(self, lambda_trans=0.1):
        super().__init__()
        self.lambda_trans = lambda_trans
    
    def forward(self, pred, target, weights=None):
        """
        Compute Translation Centered Loss (L_TC)
        
        This implementation achieves gradient centering by decomposing the loss:
        
        Mathematical equivalence:
        - Standard: L_FM = (1/B) Σ ||pred_i - target_i||^2
        - Gradient: G = ∇ L_FM
        
        - Centered: L_TC = (1/B) Σ ||pred_i - target_i - (pred̄ - target̄)||^2 + λ_trans · ||pred̄ - target̄||^2
        - Gradient: G̃ = ∇ L_TC = G - (1 - λ_trans) · Ḡ
        
        This loss decomposition is mathematically equivalent to gradient centering
        because the gradient of the centered loss automatically applies centering.
        
        Args:
            pred: Predicted values [B, D] where D is dimension (1 or 2)
            target: Target values [B, D]
            weights: Optional sample weights [B,] for UOTRFM
            
        Returns:
            loss_tc: Translation Centered loss (L_TC)
            logs: Dictionary with loss components
        """
        # Error map
        error = pred - target  # [B, D]
        
        # Translation component: batch mean error (global shift)
        error_trans = error.mean(dim=0, keepdim=True)  # [1, D]
        
        # Structure component: individual deviations from batch mean
        error_struct = error - error_trans  # [B, D]
        
        # Compute losses
        if weights is not None:
            weights = weights.view(-1, 1)  # [B, 1]
            loss_struct = (weights * (error_struct ** 2)).mean()
            loss_trans = (error_trans ** 2).mean()  # Translation is batch-level, no weighting
        else:
            loss_struct = (error_struct ** 2).mean()
            loss_trans = (error_trans ** 2).mean()
        
        # Weighted combination: L_TC = L_struct + λ_trans · L_trans
        # This is mathematically equivalent to gradient centering:
        # ∇ L_TC = ∇ L_struct + λ_trans · ∇ L_trans
        #        = (G - Ḡ) + λ_trans · Ḡ
        #        = G - (1 - λ_trans) · Ḡ
        # where G = ∇ L_FM and Ḡ = batch mean of G
        loss_tc = loss_struct + (self.lambda_trans * loss_trans)
        
        # Also compute L_FM for logging
        if weights is not None:
            loss_fm = (weights * (error ** 2)).mean()
        else:
            loss_fm = (error ** 2).mean()
        
        return loss_tc, {
            "loss_fm": loss_fm.item(),
            "loss_struct": loss_struct.item(),
            "loss_trans": loss_trans.item(),
            "loss_total": loss_tc.item()
        }


class TDMF(nn.Module):
    """
    Translation Decoupled MeanFlow (TDMF) model
    
    Based on Improved MeanFlow (iMF) with Translation Centered Loss (L_TC).
    
    Key improvements:
    1. v-loss reformulation from iMF (ground-truth target)
    2. Gradient Spatial Centering to prevent easy translation learning
    
    Mathematical foundation:
    - Standard loss: L_FM = ||V - v_target||^2
    - Gradient centering: G̃ = G - (1 - λ_trans) · Ḡ
    - This implements L_TC (Translation Centered Loss)
    
    Args:
        input_dim: Dimension of input space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
        lambda_trans: Weight for translation gradient suppression (default: 0.1)
            - 0.0: Complete gradient centering (removes all translation)
            - 1.0: Standard Flow Matching (no centering)
            - 0.1: Partial centering (90% translation suppression)
        lambda_schedule: Schedule type ('fixed' or 'linear')
    """
    
    def __init__(self, input_dim=1, hidden_dims=None, lambda_trans=0.1, lambda_schedule='fixed'):
        super(TDMF, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 32]
        
        self.input_dim = input_dim
        self.lambda_trans = lambda_trans
        self.lambda_schedule = lambda_schedule
        self._current_lambda = lambda_trans  # For scheduling
        
        # Mean velocity network u_θ(z, t, r)
        # Input: [z, t, r] concatenated
        self.mean_velocity_net = BaseMLP(
            input_dim=input_dim + 2,  # z + t + r
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )
        
        # Translation decoupled loss module
        self.td_loss = TranslationDecoupledLoss(lambda_trans=lambda_trans)
    
    def update_lambda(self, epoch, total_epochs):
        """
        Update lambda_trans based on schedule
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
        """
        if self.lambda_schedule == 'fixed':
            self._current_lambda = self.lambda_trans
        elif self.lambda_schedule == 'linear':
            # Linear schedule: lambda_trans -> 1.0
            progress = epoch / max(total_epochs - 1, 1)
            self._current_lambda = self.lambda_trans + (1.0 - self.lambda_trans) * progress
        
        self.td_loss.lambda_trans = self._current_lambda
    
    def forward(self, z, t, r):
        """
        Compute mean velocity field u_θ(z, t, r)
        
        Args:
            z: Position (batch_size, input_dim)
            t: Current time (batch_size, 1)
            r: Reference time (batch_size, 1)
            
        Returns:
            u: Mean velocity field (batch_size, input_dim)
        """
        input_vec = torch.cat([z, t, r], dim=-1)
        u = self.mean_velocity_net(input_vec)
        return u
    
    def compute_instantaneous_velocity(self, z, t, r):
        """
        Compute instantaneous velocity v from mean velocity
        For inference, use simplified approximation: v ≈ u
        
        Args:
            z: Position (batch_size, input_dim)
            t: Current time (batch_size, 1)
            r: Reference time (batch_size, 1)
            
        Returns:
            v: Instantaneous velocity field (batch_size, input_dim)
        """
        return self.forward(z, t, r)
    
    def loss_function(self, x, t, r, e=None, weights=None):
        """
        Compute TDMF loss using v-loss reformulation with Translation Centered Loss (L_TC)
        
        Mathematical formulation:
        
        Step 1: Compute compound velocity V
            e ~ N(0, I) or provided noise
            z = (1 - t) · x + t · e  (interpolated position)
            v_target = e - x  (ground-truth conditional velocity)
            
            v_pred = u_θ(z, t, t)  (predicted velocity at r=t)
            u, dudt = JVP(u_θ, (z, r, t), (v_pred, 0, 1))
            V = u + (t - r) · stopgrad(dudt)
        
        Step 2: Compute Translation Centered Loss (L_TC)
            L_FM = (1/B) Σ_{i=1}^B ||V_i - v_target_i||^2  (standard Flow Matching loss)
            
            During backward pass, gradient centering hook applies:
            G_i = ∇_{u_θ} L_FM  (gradient for sample i)
            Ḡ = (1/B) Σ_{i=1}^B G_i  (batch mean gradient)
            G̃_i = G_i - (1 - λ_trans) · Ḡ  (centered gradient)
            
            This implements L_TC: optimization uses G̃ instead of G
        
        Key properties:
        - When λ_trans = 0: Complete gradient centering (Σ G̃_i = 0)
        - When λ_trans = 1: Standard Flow Matching (G̃ = G)
        - When 0 < λ_trans < 1: Partial centering
        
        Args:
            x: Data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            r: Reference time (batch_size, 1)
            e: Noise samples (batch_size, input_dim), optional (for CFM coupling)
            weights: Optional loss weights for UOTRFM (batch_size,)
            
        Returns:
            loss: Translation Centered Loss (L_TC) with gradient hook applied
            logs: Dictionary with loss components
        """
        # Sample noise (or use provided noise for CFM coupling)
        if e is None:
            e = torch.randn_like(x)
        
        # Compute interpolated position: z = (1-t) * x + t * e
        z = (1 - t) * x + t * e
        
        # Ground-truth conditional velocity: v_target = e - x
        v_target = e - x
        
        # Step 1: Get predicted instantaneous velocity at r=t (where u = v)
        with torch.no_grad():
            v_pred = self.forward(z, t, t)
        
        # Step 2: Compute JVP using predicted velocity
        def model_fn(z_in, r_in, t_in):
            return self.forward(z_in, t_in, r_in)
        
        primals = (z, r, t)
        tangents = (v_pred, torch.zeros_like(r), torch.ones_like(t))
        
        # Compute u and JVP result
        u, jvp_result = torch.func.jvp(model_fn, primals, tangents)
        dudt = jvp_result
        
        # Step 3: Compute compound function V
        V = u + (t - r) * dudt.detach()
        
        # Step 4: Translation decoupled loss
        loss, logs = self.td_loss(V, v_target, weights=weights)
        
        return loss, logs
    
    def sample_mean_velocity(self, z, device='cpu'):
        """
        Sample using mean velocity (one-step generation)
        
        For MeanFlow, we use:
        x = z - (1.0 - 0.0) * u_θ(z, t=1.0, r=0.0)
        
        Args:
            z: Initial samples from N(0, I) (n_samples, input_dim)
            device: Device to run on
            
        Returns:
            x_pred: Predicted samples (n_samples, input_dim)
        """
        self.eval()
        
        n_samples = z.shape[0]
        
        with torch.no_grad():
            t_one = torch.ones(n_samples, 1).to(device)
            r_zero = torch.zeros(n_samples, 1).to(device)
            u = self.forward(z, t_one, r_zero)
            
            # One-step prediction: x = z - (t - r) * u
            x_pred = z - (t_one - r_zero) * u
        
        return x_pred
    
    def sample(self, z_init, n_steps=100, device='cpu'):
        """
        Sample using ODE solver with mean velocity (multi-step generation)
        
        Time goes from 1.0 to 0.0 (noise → data)
        Update: z = z - (t - r) * u
        
        Args:
            z_init: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (default: 100)
            device: Device to run on
            
        Returns:
            trajectory: Full trajectory from t=1 to t=0 (n_steps+1, n_samples, input_dim)
        """
        self.eval()
        
        n_samples = z_init.shape[0]
        
        # Time goes from 1.0 to 0.0
        t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        
        # Store trajectory
        trajectory = torch.zeros(n_steps + 1, n_samples, self.input_dim).to(device)
        trajectory[0] = z_init
        
        z = z_init.clone()
        
        with torch.no_grad():
            for step in range(n_steps):
                t_curr = t_vals[step]
                r_curr = t_vals[step + 1]
                
                t_tensor = torch.ones(n_samples, 1).to(device) * t_curr
                r_tensor = torch.ones(n_samples, 1).to(device) * r_curr
                
                u = self.forward(z, t_tensor, r_tensor)
                z = z - (t_curr - r_curr) * u
                
                trajectory[step + 1] = z
        
        return trajectory
    
    def sample_mean_velocity_ode(self, z_init, n_steps=100, device='cpu'):
        """
        Sample using ODE solver with mean velocity u only (multi-step generation)
        Same as sample() - kept for API compatibility
        
        Args:
            z_init: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (default: 100)
            device: Device to run on
            
        Returns:
            trajectory: Full trajectory from t=1 to t=0 (n_steps+1, n_samples, input_dim)
        """
        return self.sample(z_init, n_steps, device)


if __name__ == "__main__":
    # Test TDMF
    print("Testing Translation Decoupled MeanFlow (TDMF)...")
    
    # Test 1D
    print("\n=== 1D Test ===")
    model_1d = TDMF(input_dim=1, lambda_trans=0.1, lambda_schedule='fixed')
    print(f"Model architecture:\n{model_1d}")
    
    batch_size = 10
    z = torch.randn(batch_size, 1)
    t = torch.rand(batch_size, 1)
    r = torch.rand(batch_size, 1)
    u = model_1d(z, t, r)
    print(f"\nMean velocity u shape: {u.shape}")
    
    # Test loss
    x = torch.randn(batch_size, 1)
    loss, logs = model_1d.loss_function(x, t, r)
    print(f"\nLoss (L_TC): {loss.item():.4f}")
    print(f"  Flow Matching loss (L_FM): {logs['loss_fm']:.4f}")
    print(f"  Structure component: {logs['loss_struct']:.4f}")
    print(f"  Translation component: {logs['loss_trans']:.4f}")
    
    # Test sampling
    z_test = torch.randn(5, 1)
    x_pred = model_1d.sample_mean_velocity(z_test)
    print(f"\nOne-step sampling shape: {x_pred.shape}")
    
    trajectory = model_1d.sample(z_test, n_steps=10)
    print(f"Multi-step trajectory shape: {trajectory.shape}")
    
    # Test 2D
    print("\n=== 2D Test ===")
    model_2d = TDMF(input_dim=2, lambda_trans=0.1, lambda_schedule='linear')
    
    z_2d = torch.randn(batch_size, 2)
    t_2d = torch.rand(batch_size, 1)
    r_2d = torch.rand(batch_size, 1)
    u_2d = model_2d(z_2d, t_2d, r_2d)
    print(f"Mean velocity u shape (2D): {u_2d.shape}")
    
    x_2d = torch.randn(batch_size, 2)
    loss_2d, logs_2d = model_2d.loss_function(x_2d, t_2d, r_2d)
    print(f"\n2D Loss (L_TC): {loss_2d.item():.4f}")
    print(f"  Flow Matching loss (L_FM): {logs_2d['loss_fm']:.4f}")
    print(f"  Structure component: {logs_2d['loss_struct']:.4f}")
    print(f"  Translation component: {logs_2d['loss_trans']:.4f}")
    
    # Test lambda scheduling
    print("\n=== Lambda Scheduling Test ===")
    model_sched = TDMF(input_dim=1, lambda_trans=0.1, lambda_schedule='linear')
    print(f"Initial lambda: {model_sched._current_lambda}")
    
    for epoch in [0, 50, 100]:
        model_sched.update_lambda(epoch, 100)
        print(f"Epoch {epoch}: lambda = {model_sched._current_lambda:.3f}")
    
    n_params = sum(p.numel() for p in model_1d.parameters())
    print(f"\nTotal parameters: {n_params}")
