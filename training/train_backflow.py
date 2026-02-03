"""
Training script for BackFlow model
Implements the exact algorithm from original CIFAR10 implementation
"""
import torch
import torch.nn.functional as F
from torch.func import jvp
import numpy as np
from tqdm import tqdm


# ==============================================================================
# Configuration (matching original)
# ==============================================================================
CONFIG = {
    "logit_normal_mu": -2.0,
    "logit_normal_sigma": 2.0,
    "r_neq_t_ratio": 0.75,
    "p_loss": 0.75,
}


# ==============================================================================
# Core Algorithm Functions (EXACT from original)
# ==============================================================================
def prepare_r_t(batch_size, device):
    """
    Sample r and t from logit-normal distribution.
    This is the EXACT function from original code (lines 231-243).
    """
    mu, sigma = CONFIG["logit_normal_mu"], CONFIG["logit_normal_sigma"]
    t = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)
    r = torch.sigmoid(torch.randn(batch_size, device=device) * sigma + mu)

    # Original code has this commented out, so we keep it commented
    # mask = r > t
    # r_new = torch.where(mask, t, r)
    # t_new = torch.where(mask, r, t)
    # r, t = r_new, t_new

    # With probability (1 - r_neq_t_ratio), set r = t
    mask_eq = torch.rand(batch_size, device=device) > CONFIG["r_neq_t_ratio"]
    r = torch.where(mask_eq, t, r)
    return r, t


def compute_imf_loss(model, x):
    """
    Compute the Instantaneous Mean Flow (IMF) loss.
    This is the EXACT algorithm from original code (lines 246-278).

    The key steps are:
    1. Sample r, t from logit-normal
    2. Create interpolated point z_t = (1-t)*x + t*e
    3. Compute target velocity v_target = e - x
    4. Use JVP to compute du/dt
    5. Compute V_theta = u_theta + (t - r) * du/dt
    6. Use weighted MSE loss
    """
    device = x.device
    B = x.shape[0]

    # Sample r, t
    r, t = prepare_r_t(B, device)

    # Sample noise
    e = torch.randn_like(x)

    # Interpolate: z_t = (1-t)*x + t*e
    t_broad = t.view(B, 1)
    r_broad = r.view(B, 1)
    z_t = (1 - t_broad) * x + t_broad * e

    # Target velocity
    v_target = e - x

    # Compute du/dt using JVP (Jacobian-vector product)
    with torch.no_grad():
        model.eval()
        # First compute v_pred = model(z_t, t, t)
        v_pred = model(z_t, t, t)

        # Define model function for JVP
        def model_fn(z, r_arg, t_arg):
            return model(z, r_arg, t_arg)

        # JVP: tangent vectors for (z, r, t)
        # We want d/dt, so tangent for t is 1, others are 0 or v_pred
        tangents = (v_pred, torch.zeros_like(r), torch.ones_like(t))
        _, dudt = jvp(model_fn, (z_t, r, t), tangents)

        model.train()

    # Compute u_theta = model(z_t, r, t)
    u_theta = model(z_t, r, t)

    # Compute V_theta = u_theta + (t - r) * du/dt
    V_theta = u_theta + (t_broad - r_broad) * dudt.detach()

    # Compute loss with weighting
    diff_sq = (V_theta - v_target) ** 2
    loss_sum = torch.sum(diff_sq, dim=1)

    # Weighted loss (same as original)
    c = 1e-3
    p = CONFIG["p_loss"]
    w = 1 / (loss_sum + c).pow(p)
    loss = (w.detach() * loss_sum).mean()

    return loss


# ==============================================================================
# Training Function
# ==============================================================================
def train_backflow(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
                   viz_freq=200, save_dir='/home/user/Desktop/Gen_Study/outputs', dim='1d'):
    """
    Train BackFlow model.

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        seed: Random seed
        device: Device to train on
        viz_freq: Frequency to save visualizations (every N epochs)
        save_dir: Directory to save visualizations
        dim: Dimension ('1d' or '2d')

    Returns:
        model: Trained model
        losses: List of losses
    """
    import os
    from data.synthetic import generate_data, generate_data_2d, sample_prior
    from models.backflow import BackFlow

    if dim == '1d':
        from visualization.viz_backflow import visualize_backflow, compute_trajectories, one_step_decode
    else:
        from visualization.viz_backflow_2d import visualize_backflow_2d

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine input dimension
    input_dim = 1 if dim == '1d' else 2

    # Generate data
    if dim == '1d':
        x_data = generate_data(n_samples=n_samples, seed=seed)
        x_tensor = torch.FloatTensor(x_data).unsqueeze(1).to(device)
    else:  # 2d
        x_data = generate_data_2d(n_samples=n_samples, seed=seed)
        x_tensor = torch.FloatTensor(x_data).to(device)

    # Create model
    model = BackFlow(input_dim=input_dim).to(device)

    # Optimizer (same as original)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    losses = []

    # Create dataset
    dataset = torch.utils.data.TensorDataset(x_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Fixed inference samples for visualization
    n_infer = 200
    z_infer = sample_prior(n_samples=n_infer, seed=seed + 1000, dim=input_dim)
    if dim == '1d':
        z_infer_tensor = torch.FloatTensor(z_infer).unsqueeze(1).to(device)
    else:
        z_infer_tensor = torch.FloatTensor(z_infer).to(device)

    # Create visualization directory
    os.makedirs(save_dir, exist_ok=True)

    print("Training BackFlow...")
    pbar = tqdm(range(epochs), desc="Training")

    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for (batch_x,) in dataloader:
            # Compute loss using EXACT algorithm
            loss = compute_imf_loss(model, batch_x)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (same as original)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Generate visualization every viz_freq epochs
        if (epoch + 1) % viz_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                if dim == '1d':
                    # Compute ODE trajectories using Euler method
                    trajectories = compute_trajectories(model, z_infer_tensor, n_steps=100)

                    # One-step decode
                    onestep_final = one_step_decode(model, z_infer_tensor).squeeze().cpu().numpy()

                    # Save visualization
                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_backflow(
                        trajectories=trajectories,
                        onestep_final=onestep_final,
                        save_path=viz_path,
                        epoch=epoch + 1,
                        x_data=x_data
                    )
                else:  # 2d
                    from visualization.viz_backflow_2d import compute_trajectories_2d, one_step_decode_2d

                    # Compute ODE trajectories
                    trajectories = compute_trajectories_2d(model, z_infer_tensor, steps=100)

                    # One-step decode
                    x_generated_onestep = one_step_decode_2d(model, z_infer_tensor).cpu().numpy()

                    # Save visualization
                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_backflow_2d(
                        trajectories=trajectories,
                        onestep_samples=x_generated_onestep,
                        x_data=x_data,
                        save_path=viz_path,
                        epoch=epoch + 1
                    )

    print(f"Training complete. Final loss: {losses[-1]:.6f}")

    return model, losses
