"""
Flow Matching model
"""
import torch
import torch.nn as nn
from models.base_mlp import BaseMLP


class FlowMatching(nn.Module):
    """
    Flow Matching model using Conditional Flow Matching (CFM)

    Training:
        Learn velocity field v_θ(x_t, t) that matches the conditional flow
        x_t = (1-t) * z + t * x where z ~ N(0, I), x ~ p_data

    Inference:
        Solve ODE: dx/dt = v_θ(x_t, t) from t=0 to t=1
        Starting from z ~ N(0, I) at t=0

    Args:
        input_dim: Dimension of input space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
    """

    def __init__(self, input_dim=1, hidden_dims=None):
        super(FlowMatching, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim

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
            t: Time (batch_size, 1)

        Returns:
            v: Velocity field (batch_size, input_dim)
        """
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

    def loss_function(self, z, x, t, weights=None):
        """
        Compute Flow Matching loss: ||v_θ(x_t, t) - u_t||^2

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
            # Weighted loss for UOTRFM
            # weights shape: (batch_size,), need to reshape for broadcasting
            weights = weights.view(-1, 1)  # (batch_size, 1)
            loss = (weights * (v_pred - u_t) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(v_pred, u_t, reduction='mean')

        return loss

    def sample(self, z, n_steps=100, device='cpu'):
        """
        Sample from the model using Euler ODE solver

        Args:
            z: Initial samples from N(0, I) (n_samples, input_dim)
            n_steps: Number of ODE solver steps (default: 100)
            device: Device to run on

        Returns:
            trajectory: Full trajectory from t=0 to t=1 (n_steps+1, n_samples, input_dim)
        """
        self.eval()

        n_samples = z.shape[0]
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
    # Test Flow Matching
    print("Testing Flow Matching...")

    # Create model
    model = FlowMatching(input_dim=1)
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x_t = torch.randn(batch_size, 1)
    t = torch.rand(batch_size, 1)
    v = model(x_t, t)
    print(f"\nInput x_t shape: {x_t.shape}")
    print(f"Time t shape: {t.shape}")
    print(f"Velocity shape: {v.shape}")

    # Test conditional flow
    z = torch.randn(batch_size, 1)
    x = torch.randn(batch_size, 1)
    x_t, u_t = model.compute_conditional_flow(z, x, t)
    print(f"\nConditional flow x_t shape: {x_t.shape}")
    print(f"Conditional velocity u_t shape: {u_t.shape}")

    # Test loss computation
    loss = model.loss_function(z, x, t)
    print(f"\nFlow matching loss: {loss.item():.4f}")

    # Test sampling
    z_test = torch.randn(5, 1)
    trajectory = model.sample(z_test, n_steps=100)
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"Initial position (t=0): {trajectory[0].squeeze().numpy()}")
    print(f"Final position (t=1): {trajectory[-1].squeeze().numpy()}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params}")
