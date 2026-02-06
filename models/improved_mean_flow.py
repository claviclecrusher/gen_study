"""
Improved MeanFlow (iMF) model - v-loss reformulation
Reference: https://arxiv.org/html/2512.02012v1

Key improvement over original MeanFlow:
- Original MF: loss = ||u - u_tgt||² where u_tgt = v - (t-r)*dudt (network-dependent target)
- Improved MF: loss = ||V - v_target||² where V = u + (t-r)*stopgrad(dudt) (ground-truth target)

This reformulation makes the regression target independent of the network,
yielding a more standard regression problem and improving training stability.
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class ImprovedMeanFlow(nn.Module):
    """
    Improved MeanFlow model using v-loss reformulation

    Key idea:
    - Learn mean velocity field u_θ(x_t, t, r)
    - Compute compound function V = u + (t - r) * stopgrad(du/dt)
    - Train with v-loss: ||V - v_target||² where v_target = e - x (ground-truth)

    The key difference from original MeanFlow:
    - Original: u_tgt = v - (t-r)*dudt, loss = ||u - u_tgt||² (target depends on network)
    - Improved: V = u + (t-r)*dudt, loss = ||V - v_target||² (target is ground-truth)

    Additionally, the JVP tangent uses predicted velocity instead of ground-truth:
    - Original: jvp tangent = (v, 1, 0) where v = e - x (unknown ground-truth)
    - Improved: jvp tangent = (v_pred, 1, 0) where v_pred is network prediction

    This ensures the input to the compound function depends only on z, not on unknown quantities.

    Args:
        input_dim: Dimension of input space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, input_dim=1, hidden_dims=None):
        super(ImprovedMeanFlow, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim

        # Mean velocity network u_θ(z, t, r)
        # Input: [z, t, r] concatenated
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
        input_vec = torch.cat([z, t, r], dim=-1)
        u = self.mean_velocity_net(input_vec)
        return u

    def compute_instantaneous_velocity(self, z, t, r):
        """
        Compute instantaneous velocity v from mean velocity

        In MeanFlow, the relationship is:
        v = u + (t - r) * du/dt

        For inference, we use simplified approximation: v ≈ u

        Args:
            z: Position (batch_size, input_dim)
            t: Current time (batch_size, 1)
            r: Reference time (batch_size, 1)

        Returns:
            v: Instantaneous velocity field (batch_size, input_dim)
        """
        # For inference, use simplified approximation
        return self.forward(z, t, r)

    def loss_function(self, x, t, r, e=None, weights=None):
        """
        Compute Improved MeanFlow loss using v-loss reformulation

        Key improvement: The regression target is now ground-truth v = e - x,
        independent of the network. This is achieved by:
        1. Computing compound function V = u + (t - r) * stopgrad(dudt)
        2. Using v_target = e - x as the target (ground-truth)

        Algorithm (from improved MeanFlow paper):
            e = randn_like(x)
            z = (1 - t) * x + t * e
            v_target = e - x  # ground-truth conditional velocity

            # Get predicted velocity at r=t (where u = v)
            v_pred = fn(z, t, t)

            # Compute u and dudt using predicted velocity in JVP
            u, dudt = jvp(fn, (z, r, t), (v_pred, 0, 1))

            # Compute compound function V
            V = u + (t - r) * stopgrad(dudt)

            # v-loss: regression to ground-truth
            loss = ||V - v_target||²

        Args:
            x: Data samples (batch_size, input_dim)
            t: Time (batch_size, 1)
            r: Reference time (batch_size, 1)
            e: Noise samples (batch_size, input_dim), optional (for CFM coupling)
            weights: Optional loss weights for UOTRFM (batch_size,)

        Returns:
            loss: v-loss (regression to ground-truth velocity)
        """
        # Sample noise (or use provided noise for CFM coupling)
        if e is None:
            e = torch.randn_like(x)

        # Compute interpolated position: z = (1-t) * x + t * e
        z = (1 - t) * x + t * e

        # Ground-truth conditional velocity: v_target = e - x
        v_target = e - x

        # Step 1: Get predicted instantaneous velocity at r=t (where u = v)
        # This avoids using ground-truth v in the JVP tangent
        with torch.no_grad():
            v_pred = self.forward(z, t, t)  # at r=t, u equals instantaneous v

        # Step 2: Compute JVP using predicted velocity
        # JVP tangents: (v_pred, 0, 1) gives us ∂u/∂z·v_pred + ∂u/∂t
        def model_fn(z_in, r_in, t_in):
            return self.forward(z_in, t_in, r_in)

        primals = (z, r, t)
        tangents = (v_pred, torch.zeros_like(r), torch.ones_like(t))

        # Compute u and JVP result (∂u/∂z·v_pred + ∂u/∂t)
        u, jvp_result = torch.func.jvp(model_fn, primals, tangents)

        # The JVP result approximates dudt (time derivative component)
        dudt = jvp_result

        # Step 3: Compute compound function V
        # V = u + (t - r) * stopgrad(dudt)
        V = u + (t - r) * dudt.detach()  # stopgrad on dudt

        # Step 4: v-loss (regression to ground-truth, optionally weighted)
        if weights is not None:
            weights = weights.view(-1, 1)  # (batch_size, 1)
            loss = (weights * (V - v_target) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(V, v_target, reduction='mean')

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
        Sample using ODE solver with mean velocity (multi-step generation)

        Time goes from 1.0 to 0.0 (noise → data)
        Update: z = z - (t - r) * u
        r = next time step value

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
                r_curr = t_vals[step + 1]  # r = next time step

                # Convert to tensors
                t_tensor = torch.ones(n_samples, 1).to(device) * t_curr
                r_tensor = torch.ones(n_samples, 1).to(device) * r_curr

                # Compute mean velocity u
                u = self.forward(z, t_tensor, r_tensor)

                # Update rule: z = z - (t - r) * u
                z = z - (t_curr - r_curr) * u

                trajectory[step + 1] = z

        return trajectory

    def sample_mean_velocity_ode(self, z_init, n_steps=100, device='cpu'):
        """
        Sample using ODE solver with mean velocity u only (multi-step generation)

        Same as sample() - kept for API compatibility with original MeanFlow

        Args:
            z_init: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (default: 100)
            device: Device to run on

        Returns:
            trajectory: Full trajectory from t=1 to t=0 (n_steps+1, n_samples, input_dim)
        """
        return self.sample(z_init, n_steps, device)


if __name__ == "__main__":
    # Test Improved MeanFlow
    print("Testing Improved MeanFlow (iMF)...")

    # Create model
    model = ImprovedMeanFlow(input_dim=1)
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
    print(f"\nImproved MeanFlow v-loss: {loss.item():.4f}")

    # Test one-step sampling (mean velocity)
    z_test = torch.randn(5, 1)
    x_pred = model.sample_mean_velocity(z_test)
    print(f"\nOne-step sampling (mean velocity):")
    print(f"  Initial z: {z_test.squeeze().numpy()}")
    print(f"  Predicted x: {x_pred.squeeze().numpy()}")

    # Test multi-step sampling (ODE trajectory)
    trajectory = model.sample(z_test, n_steps=100)
    print(f"\nMulti-step sampling (ODE trajectory):")
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Initial position (t=1): {trajectory[0].squeeze().numpy()}")
    print(f"  Final position (t=0): {trajectory[-1].squeeze().numpy()}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params}")
