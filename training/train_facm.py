"""
Training script for FACM model (1D/2D).
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import generate_data, generate_data_2d, sample_prior
from models.facm import FACM, FACMConfig


def train_facm(
    n_samples: int = 500,
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
    viz_interval: Optional[int] = None,
    viz_output_dir: Optional[str] = None,
    n_infer: int = 200,
    dim: str = "1d",
    loss_config: Optional[FACMConfig] = None,
    sampler_steps: int = 100,
    sampler_heun: bool = False,
    timestep_shift: float = 0.0,
):
    """
    Train FACM model.

    Returns:
        model, history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("Training FACM Model")
    print("=" * 60)

    input_dim = 1 if dim == "1d" else 2

    print(f"Generating {n_samples} synthetic data samples ({dim})...")
    if dim == "1d":
        x_data = generate_data(n_samples=n_samples, seed=seed)
        x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)
    else:
        x_data = generate_data_2d(n_samples=n_samples, seed=seed)
        x_data = torch.FloatTensor(x_data).to(device)

    model = FACM(input_dim=input_dim).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_config is None:
        loss_config = FACMConfig()

    # Prepare visualization data if needed
    if viz_interval is not None and viz_output_dir is not None:
        if dim == "1d":
            from visualization.viz_facm import visualize_facm
        else:
            from visualization.viz_facm_2d import visualize_facm_2d

        os.makedirs(viz_output_dir, exist_ok=True)

        z_train_viz = sample_prior(n_samples=n_samples, seed=seed, dim=input_dim)
        x_train_viz = generate_data(n_samples=n_samples, seed=seed) if dim == "1d" else generate_data_2d(n_samples=n_samples, seed=seed)
        coupling_indices_viz = np.random.permutation(n_samples)

        z_infer_viz = sample_prior(n_samples=n_infer, seed=seed + 1, dim=input_dim)

    history = {"loss": [], "cm_loss": [], "fm_loss": []}
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nTraining for {epochs} epochs...")
    if viz_interval is not None:
        print(f"Saving visualization every {viz_interval} epochs to {viz_output_dir}")

    for epoch in range(epochs):
        model.train()
        epoch_total = 0.0
        epoch_cm = 0.0
        epoch_fm = 0.0

        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            bs = end_idx - start_idx

            x_batch = x_shuffled[start_idx:end_idx]

            optimizer.zero_grad()
            total, cm, fm = model.loss_function(x_batch, config=loss_config)
            total.backward()
            optimizer.step()

            epoch_total += total.item() * bs
            epoch_cm += cm.item() * bs
            epoch_fm += fm.item() * bs

        epoch_total /= n_samples
        epoch_cm /= n_samples
        epoch_fm /= n_samples
        history["loss"].append(epoch_total)
        history["cm_loss"].append(epoch_cm)
        history["fm_loss"].append(epoch_fm)

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_total:.6f} (CM {epoch_cm:.6f}, FM {epoch_fm:.6f})"
            )

        # Intermediate visualization
        if viz_interval is not None and viz_output_dir is not None and (epoch + 1) % viz_interval == 0:
            print(f"  Saving visualization at epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                if dim == "1d":
                    z_tensor = torch.FloatTensor(z_infer_viz).unsqueeze(1).to(device)
                    traj = model.sample_euler(
                        z_tensor,
                        n_steps=sampler_steps,
                        heun=sampler_heun,
                        timestep_shift=timestep_shift,
                    ).squeeze().cpu().numpy()
                    cm_onestep = model.sample_consistency(z_tensor, n_steps=1).squeeze().cpu().numpy()

                    viz_path = os.path.join(viz_output_dir, f"epoch_{epoch+1:04d}.png")
                    visualize_facm(
                        z_samples=z_train_viz,
                        x_data=x_train_viz,
                        trajectories=traj,
                        cm_onestep=cm_onestep,
                        coupling_indices=coupling_indices_viz,
                        save_path=viz_path,
                    )
                else:
                    z_tensor = torch.FloatTensor(z_infer_viz).to(device)
                    traj = model.sample_euler(
                        z_tensor,
                        n_steps=sampler_steps,
                        heun=sampler_heun,
                        timestep_shift=timestep_shift,
                    ).cpu().numpy()
                    cm_onestep = model.sample_consistency(z_tensor, n_steps=1).cpu().numpy()

                    viz_path = os.path.join(viz_output_dir, f"epoch_{epoch+1:04d}.png")
                    visualize_facm_2d(
                        trajectories=traj,
                        cm_onestep=cm_onestep,
                        z_samples=z_infer_viz,
                        x_data=x_train_viz,
                        save_path=viz_path,
                        epoch=epoch + 1,
                    )
            model.train()

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f} (CM {history['cm_loss'][-1]:.6f}, FM {history['fm_loss'][-1]:.6f})")
    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FACM model (1D/2D)")
    parser.add_argument("--dim", type=str, default="1d", choices=["1d", "2d"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/home/user/Desktop/Gen_Study/outputs")
    parser.add_argument("--viz_interval", type=int, default=None)
    parser.add_argument("--n_infer", type=int, default=200)
    parser.add_argument("--sampler_steps", type=int, default=100)
    parser.add_argument("--heun", action="store_true")
    parser.add_argument("--timestep_shift", type=float, default=0.0)

    # Loss config knobs
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=1e-3)
    parser.add_argument("--t_type", type=str, default="uniform", choices=["uniform", "log", "default"])
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--no_cosine", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg = FACMConfig(
        p=args.p,
        c=args.c,
        t_type=args.t_type,
        mean=args.mean,
        std=args.std,
        use_cosine_in_fm=not args.no_cosine,
    )

    model, _ = train_facm(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        viz_interval=args.viz_interval,
        viz_output_dir=args.output_dir if args.viz_interval else None,
        n_infer=args.n_infer,
        dim=args.dim,
        loss_config=cfg,
        sampler_steps=args.sampler_steps,
        sampler_heun=args.heun,
        timestep_shift=args.timestep_shift,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_name = "facm_model.pt" if args.dim == "1d" else "facm_2d_model.pt"
    save_path = os.path.join(args.output_dir, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

