"""
Synthetic data generator for 1D and 2D Gaussian mixtures
"""
import numpy as np


# 1D Gaussian parameters
MEAN_1 = -2.0
MEAN_2 = 2.0
STD_1 = 0.8  # var = 0.64 < 1
STD_2 = 0.6  # var = 0.36 < 1
MIX_RATIO = 0.7  # 70% from Gaussian 1, 30% from Gaussian 2

# 2D Gaussian parameters
MEAN_1_2D = np.array([-2.0, -2.0])
MEAN_2_2D = np.array([2.0, 2.0])
COV_1_2D = np.array([[0.5, 0.0], [0.0, 0.5]])  # Independent
COV_2_2D = np.array([[0.3, 0.15], [0.15, 0.3]])  # Some correlation
MIX_RATIO_2D = 0.6  # 60% from Gaussian 1, 40% from Gaussian 2


def generate_data(n_samples=500, seed=None):
    """
    Generate 1D samples from 2-Gaussian mixture

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (n_samples,) containing 1D samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine number of samples from each Gaussian
    n_samples_1 = int(n_samples * MIX_RATIO)
    n_samples_2 = n_samples - n_samples_1

    # Generate samples from each Gaussian
    samples_1 = np.random.normal(MEAN_1, STD_1, n_samples_1)
    samples_2 = np.random.normal(MEAN_2, STD_2, n_samples_2)

    # Combine and shuffle
    samples = np.concatenate([samples_1, samples_2])
    np.random.shuffle(samples)

    return samples


def sample_prior(n_samples=500, seed=None, dim=1):
    """
    Sample from prior distribution z ~ N(0, I)

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        dim: Dimensionality (1 or 2)

    Returns:
        numpy array of shape (n_samples,) for 1D or (n_samples, 2) for 2D
    """
    if seed is not None:
        np.random.seed(seed)

    if dim == 1:
        return np.random.normal(0, 1, n_samples)
    elif dim == 2:
        return np.random.normal(0, 1, (n_samples, 2))
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def generate_data_2d(n_samples=500, seed=None):
    """
    Generate 2D samples from 2-Gaussian mixture

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        numpy array of shape (n_samples, 2) containing 2D samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine number of samples from each Gaussian
    n_samples_1 = int(n_samples * MIX_RATIO_2D)
    n_samples_2 = n_samples - n_samples_1

    # Generate samples from each Gaussian
    samples_1 = np.random.multivariate_normal(MEAN_1_2D, COV_1_2D, n_samples_1)
    samples_2 = np.random.multivariate_normal(MEAN_2_2D, COV_2_2D, n_samples_2)

    # Combine and shuffle
    samples = np.vstack([samples_1, samples_2])
    np.random.shuffle(samples)

    return samples


if __name__ == "__main__":
    # Test data generation
    import matplotlib.pyplot as plt

    data = generate_data(n_samples=1000, seed=42)
    prior = sample_prior(n_samples=1000, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot data distribution
    axes[0].hist(data, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0].set_title('2-Gaussian Mixture Data')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(MEAN_1, color='red', linestyle='--', label=f'μ1={MEAN_1}')
    axes[0].axvline(MEAN_2, color='blue', linestyle='--', label=f'μ2={MEAN_2}')
    axes[0].legend()

    # Plot prior distribution
    axes[1].hist(prior, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1].set_title('Prior N(0, 1)')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('/home/user/Desktop/Gen_Study/outputs/data_test.png', dpi=100)
    print("Data generation test complete. Plot saved to outputs/data_test.png")
