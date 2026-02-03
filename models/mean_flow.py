"""
MeanFlow model - Mean Flows for One-step Generative Modeling
Reference: https://github.com/haidog-yaqub/MeanFlow
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class MeanFlow(nn.Module):
    """
    MeanFlow model using Mean Velocity Learning

    Key idea:
    - Learn mean velocity field u_θ(x_t, t)
    - Derive instantaneous velocity v from u and its time derivative
    - u enables one-step generation, v enables multi-step ODE trajectory

    Training:
        Learn u_θ(x_t, t) where x_t = (1-t) * z + t * x
        Target: u_target = x - z (constant conditional velocity)

    Inference:
        1. Mean velocity (one-step): x_pred = z + u_θ(z, t=0)
        2. Instantaneous velocity (ODE): solve dx/dt = v_θ(x_t, t) from t=0 to t=1
           where v_θ = u_θ + (t-r) * du_θ/dt

    Args:
        input_dim: Dimension of input space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, input_dim=1, hidden_dims=None):
        super(MeanFlow, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim

        # Mean velocity network u_θ(z, t, r)
        # Input: [z, t, r] concatenated (GitHub order)
        self.mean_velocity_net = BaseMLP(
            input_dim=input_dim + 2,  # z + t + r
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )

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
        # Concatenate z, t, r (GitHub order)
        input_vec = torch.cat([z, t, r], dim=-1)
        u = self.mean_velocity_net(input_vec)
        return u

    def compute_instantaneous_velocity(self, z, t, r):
        """
        Compute instantaneous velocity v from mean velocity

        In MeanFlow, the relationship is:
        v = u + (t - r) * du/dt

        Args:
            z: Position (batch_size, input_dim)
            t: Current time (batch_size, 1)
            r: Reference time (batch_size, 1)

        Returns:
            v: Instantaneous velocity field (batch_size, input_dim)
        """
        # For inference, use simplified approximation: v ≈ u
        # This avoids gradient computation during sampling
        if not self.training:
            return self.forward(z, t, r)

        # Full computation with gradient (for completeness)
        z_copy = z.detach().requires_grad_(True)
        t_copy = t.detach().requires_grad_(True)
        r_copy = r.detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            u_copy = self.forward(z_copy, t_copy, r_copy)

            # Compute du/dt
            dudt = torch.autograd.grad(
                outputs=u_copy,
                inputs=t_copy,
                grad_outputs=torch.ones_like(u_copy),
                create_graph=False,
                retain_graph=False
            )[0]

        # Compute mean velocity at original position
        u = self.forward(z, t, r)

        # v = u + (t - r) * du/dt
        v = u + (t - r) * dudt.detach()

        return v

    def loss_function(self, x, t, r):
        """
        Compute MeanFlow loss using JVP (Jacobian-Vector Product)

        Algorithm (from GitHub):
            e = randn_like(x)
            z = (1 - t) * x + t * e
            v = e - x
            u, dudt = jvp(fn, (z, t, r), (v, 1, 0))
            u_tgt = v - (t - r) * dudt
            loss = ||u - stopgrad(u_tgt)||^2

        Args:
            x: Data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            r: Reference time (batch_size, 1)

        Returns:
            loss: Mean velocity matching loss
        """
        # Sample noise
        e = torch.randn_like(x)

        # Compute interpolated position: z = (1-t) * x + t * e
        z = (1 - t) * x + t * e

        # Conditional velocity: v = e - x
        v = e - x

        # Use torch.func.jvp for proper JVP computation
        # JVP computes: ∂u/∂z·v + ∂u/∂t·1 + ∂u/∂r·0
        def model_fn(z_in, t_in, r_in):
            return self.forward(z_in, t_in, r_in)

        # Primals: (z, t, r)
        # Tangents: (v, 1, 0) - this gives us ∂u/∂z·v + ∂u/∂t
        primals = (z, t, r)
        tangents = (v, torch.ones_like(t), torch.zeros_like(r))

        # Compute JVP: u and (∂u/∂z·v + ∂u/∂t)
        u_pred, jvp_result = torch.func.jvp(model_fn, primals, tangents)

        # The JVP result is: ∂u/∂z·v + ∂u/∂t
        # We need to extract dudt, which is part of this result
        # According to the paper: u_tgt = v - (t - r) * dudt
        # where dudt is ∂u/∂t (the time derivative component)

        # The jvp_result contains ∂u/∂z·v + ∂u/∂t
        # To get pure ∂u/∂t, we need to compute it separately or use the full JVP result
        # Based on GitHub code, they use the full JVP result directly
        dudt = jvp_result

        # Target mean velocity: u_tgt = v - (t - r) * dudt
        u_tgt = v - (t - r) * dudt.detach()  # stopgrad on dudt

        # MSE loss
        loss = nn.functional.mse_loss(u_pred, u_tgt, reduction='mean')

        return loss

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
            # Start from t=1.0 (noise), r=0.0
            t_one = torch.ones(n_samples, 1).to(device)
            r_zero = torch.zeros(n_samples, 1).to(device)
            u = self.forward(z, t_one, r_zero)

            # One-step prediction: x = z - (t - r) * u
            x_pred = z - (t_one - r_zero) * u

        return x_pred

    def sample(self, z_init, n_steps=100, device='cpu'):
        """
        Sample using ODE solver with instantaneous velocity v (multi-step generation)

        Following GitHub implementation:
        - Time goes from 1.0 to 0.0 (noise → data)
        - Update: z = z - (t - r) * v
        - r = next time step value (NOT 0!)

        Args:
            z_init: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (default: 100)
            device: Device to run on

        Returns:
            trajectory: Full trajectory from t=1 to t=0 (n_steps+1, n_samples, input_dim)
        """
        self.eval()

        n_samples = z_init.shape[0]

        # Time goes from 1.0 to 0.0 (GitHub implementation)
        t_vals = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        # Store trajectory
        trajectory = torch.zeros(n_steps + 1, n_samples, self.input_dim).to(device)
        trajectory[0] = z_init

        z = z_init.clone()

        with torch.no_grad():
            for step in range(n_steps):
                # Current time and next time
                t_curr = t_vals[step]
                r_curr = t_vals[step + 1]  # r = next time step!

                # Convert to tensors
                t_tensor = torch.ones(n_samples, 1).to(device) * t_curr
                r_tensor = torch.ones(n_samples, 1).to(device) * r_curr

                # Compute velocity
                v = self.forward(z, t_tensor, r_tensor)

                # GitHub update rule: z = z - (t - r) * v
                # (t - r) is the step size
                z = z - (t_curr - r_curr) * v

                trajectory[step + 1] = z

        return trajectory

    def sample_mean_velocity_ode(self, z_init, n_steps=100, device='cpu'):
        """
        Sample using ODE solver with mean velocity u only (multi-step generation)

        Similar to sample() but uses mean velocity u with r = next time step

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
                # Current time and next time
                t_curr = t_vals[step]
                r_curr = t_vals[step + 1]  # r = next time step!

                # Convert to tensors
                t_tensor = torch.ones(n_samples, 1).to(device) * t_curr
                r_tensor = torch.ones(n_samples, 1).to(device) * r_curr

                # Compute mean velocity u
                u = self.forward(z, t_tensor, r_tensor)

                # Update rule: z = z - (t - r) * u
                z = z - (t_curr - r_curr) * u

                trajectory[step + 1] = z

        return trajectory


if __name__ == "__main__":
    # Test MeanFlow
    print("Testing MeanFlow...")

    # Create model
    model = MeanFlow(input_dim=1)
    print(f"Model architecture:\n{model}")

    # Test forward pass (mean velocity)
    batch_size = 10
    z = torch.randn(batch_size, 1)
    t = torch.rand(batch_size, 1)
    r = torch.rand(batch_size, 1)
    u = model(z, t, r)
    print(f"\nMean velocity u:")
    print(f"  Input z shape: {z.shape}")
    print(f"  Time t shape: {t.shape}")
    print(f"  Reference time r shape: {r.shape}")
    print(f"  Mean velocity u shape: {u.shape}")

    # Test instantaneous velocity
    v = model.compute_instantaneous_velocity(z, t, r)
    print(f"\nInstantaneous velocity v:")
    print(f"  Velocity v shape: {v.shape}")

    # Test loss computation
    x = torch.randn(batch_size, 1)
    loss = model.loss_function(x, t, r)
    print(f"\nMeanFlow loss: {loss.item():.4f}")

    # Test one-step sampling (mean velocity)
    z_test = torch.randn(5, 1)
    x_pred = model.sample_mean_velocity(z_test)
    print(f"\nOne-step sampling (mean velocity):")
    print(f"  Initial z: {z_test.squeeze().numpy()}")
    print(f"  Predicted x: {x_pred.squeeze().numpy()}")

    # Test multi-step sampling (ODE with instantaneous velocity)
    trajectory = model.sample(z_test, n_steps=100)
    print(f"\nMulti-step sampling (ODE trajectory):")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Initial position (t=0): {trajectory[0].squeeze().numpy()}")
    print(f"  Final position (t=1): {trajectory[-1].squeeze().numpy()}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params}")
