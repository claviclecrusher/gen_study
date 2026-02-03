"""
2D Synthetic data generation for flow matching experiments
"""
import numpy as np


def generate_data_2d(n_samples=500, seed=42):
    """
    Generate 2D target data from two Gaussians

    Target distribution: mixture of 2 Gaussians at [3, 1] and [3, -1]

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        data: (n_samples, 2) array of 2D samples
    """
    np.random.seed(seed)

    # Split samples between two Gaussians
    n_per_mode = n_samples // 2

    # First Gaussian: center at [3, 1]
    data1 = np.random.randn(n_per_mode, 2) * 0.3 + np.array([3.0, 1.0])

    # Second Gaussian: center at [3, -1]
    data2 = np.random.randn(n_samples - n_per_mode, 2) * 0.3 + np.array([3.0, -1.0])

    # Combine and shuffle
    data = np.vstack([data1, data2])
    np.random.shuffle(data)

    return data


def sample_prior_2d(n_samples=500, seed=42):
    """
    Sample from 2D prior distribution (single Gaussian at origin)

    Source distribution: Gaussian at [0, 0]

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        samples: (n_samples, 2) array of 2D samples
    """
    np.random.seed(seed)

    # Single Gaussian at origin [0, 0]
    samples = np.random.randn(n_samples, 2) * 0.5

    return samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test data generation
    print("Testing 2D data generation...")

    z_samples = sample_prior_2d(n_samples=500, seed=42)
    x_data = generate_data_2d(n_samples=500, seed=42)

    print(f"Source samples shape: {z_samples.shape}")
    print(f"Target samples shape: {x_data.shape}")
    print(f"Source mean: {z_samples.mean(axis=0)}")
    print(f"Target mean: {x_data.mean(axis=0)}")

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(z_samples[:, 0], z_samples[:, 1], alpha=0.5, label='Source (z)', s=10)
    plt.scatter(x_data[:, 0], x_data[:, 1], alpha=0.5, label='Target (x)', s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Source and Target Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('/home/user/Desktop/Gen_Study/outputs/test_2d_data.png', dpi=150)
    print("Test visualization saved!")
