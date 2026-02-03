"""
Base MLP module for all models to inherit from
"""
import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    """
    Base Multi-Layer Perceptron with configurable architecture

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions (default: [32, 64, 32])
        activation: Activation function (default: ReLU)
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation=nn.ReLU):
        super(BaseMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build layers
        layers = []
        prev_dim = input_dim

        # Hidden layers with activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


if __name__ == "__main__":
    # Test BaseMLP
    print("Testing BaseMLP...")

    # Create model
    model = BaseMLP(input_dim=1, output_dim=1, hidden_dims=[32, 64, 32])
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params}")
