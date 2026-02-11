"""
Training script for ModeFlowMatching (ModeFM) model

Uses Gaussian Kernel Loss instead of L2. OTCFM default. UOTRFM not supported.
"""
import torch
import torch.optim as optim
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modefm import ModeFlowMatching
from data.synthetic import generate_data, generate_data_2d, sample_prior
from utils.cfm_sampler import create_cfm_sampler


def _compute_sigma(epoch, epochs, initial_sigma, min_sigma, schedule, schedule_params):
    """
    Compute sigma for given epoch based on schedule type.

    Schedules:
        exponential: sigma = init * decay^epoch (fast early decay)
        cosine: sigma = min + (init - min) * 0.5 * (1 + cos(pi * epoch / epochs)) (slower early)
        linear: sigma = init - (init - min) * epoch / epochs
        step: sigma = init * gamma^(epoch // step_size)
        warmup_cosine: keep init for warmup_epochs, then cosine decay to min
        warmup_linear: keep init for warmup_epochs, then linear decay to min
        three_phase_linear: hold init -> linear decay -> hold min (default 1/3 each)
        sigmoid: hold init -> sigmoid decay -> hold min (default ~1/3 each)
        batch_adaptive: sigma from batch residual quantile (no schedule, computed per batch)
        sample_adaptive: sigma from var_head output (requires use_var_head)
    """
    decay_factor = schedule_params.get('decay_factor', 0.95)
    step_size = schedule_params.get('step_size', max(1, epochs // 4))
    gamma = schedule_params.get('gamma', 0.5)
    warmup_epochs = schedule_params.get('warmup_epochs') or max(1, int(epochs * schedule_params.get('warmup_ratio', 0.2)))

    if schedule == 'exponential':
        sigma = initial_sigma * (decay_factor ** epoch)
    elif schedule == 'cosine':
        # Smooth: high at start, gentle decay, steep at end
        progress = min(1.0, epoch / max(1, epochs))
        sigma = min_sigma + (initial_sigma - min_sigma) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == 'linear':
        progress = min(1.0, epoch / max(1, epochs))
        sigma = initial_sigma - (initial_sigma - min_sigma) * progress
    elif schedule == 'step':
        n_steps = epoch // max(1, step_size)
        sigma = initial_sigma * (gamma ** n_steps)
    elif schedule == 'warmup_cosine':
        if epoch < warmup_epochs:
            sigma = initial_sigma
        else:
            # Cosine from init to min over (epochs - warmup_epochs)
            decay_epochs = epochs - warmup_epochs
            progress = min(1.0, (epoch - warmup_epochs) / max(1, decay_epochs))
            sigma = min_sigma + (initial_sigma - min_sigma) * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule == 'warmup_linear':
        if epoch < warmup_epochs:
            sigma = initial_sigma
        else:
            decay_epochs = epochs - warmup_epochs
            progress = min(1.0, (epoch - warmup_epochs) / max(1, decay_epochs))
            sigma = initial_sigma - (initial_sigma - min_sigma) * progress
    elif schedule == 'three_phase_linear':
        # Phase 1: hold init, Phase 2: linear decay, Phase 3: hold min (default 1/3 each)
        hold_ratio = schedule_params.get('hold_ratio', 1.0 / 3)
        decay_ratio = schedule_params.get('decay_ratio', 1.0 / 3)
        phase1_end = max(1, int(epochs * hold_ratio))
        phase2_end = phase1_end + max(1, int(epochs * decay_ratio))
        phase2_end = min(phase2_end, epochs)  # cap at total epochs
        if epoch < phase1_end:
            sigma = initial_sigma
        elif epoch < phase2_end:
            progress = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
            sigma = initial_sigma - (initial_sigma - min_sigma) * progress
        else:
            sigma = min_sigma
    elif schedule == 'sigmoid':
        # Phase 1: hold init, Phase 2: sigmoid decay, Phase 3: hold min (default ~1/3 each)
        hold_ratio = schedule_params.get('hold_ratio', 1.0 / 3)
        decay_ratio = schedule_params.get('decay_ratio', 1.0 / 3)
        steepness = schedule_params.get('steepness', 8.0)  # k in 1/(1+exp(-k*(x-0.5)))
        phase1_end = max(1, int(epochs * hold_ratio))
        phase2_end = phase1_end + max(1, int(epochs * decay_ratio))
        phase2_end = min(phase2_end, epochs)
        if epoch < phase1_end:
            sigma = initial_sigma
        elif epoch < phase2_end:
            progress = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
            # Sigmoid: 0->0, 1->1, S-curve centered at 0.5
            sigmoid_val = 1.0 / (1.0 + math.exp(-steepness * (progress - 0.5)))
            sigma = initial_sigma + (min_sigma - initial_sigma) * sigmoid_val
        else:
            sigma = min_sigma
    elif schedule in ('batch_adaptive', 'sample_adaptive'):
        # Not used - sigma computed per batch/sample in loss_function
        sigma = initial_sigma  # Placeholder for logging
    else:
        raise ValueError(f"Unknown sigma_schedule: {schedule}")

    return max(min_sigma, sigma)


def train_modefm(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
                 viz_freq=200, save_dir='/home/user/Desktop/Gen_Study/outputs', dim='1d',
                 cfm_type='otcfm', cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
                 dataset_2d='2gauss', lr_scheduler='cosine', lr_scheduler_params=None,
                 initial_sigma=None, min_sigma=0.1, sigma_decay_factor=0.95,
                 sigma_schedule='cosine', sigma_schedule_params=None,
                 use_var_head=False, var_loss_weight=1.0, sigma_scale=1.0,
                 sample_adaptive_warmup_epochs=0, sample_adaptive_warmup_sigma=10.0):
    """
    Train ModeFlowMatching model with Gaussian Kernel Loss

    Args:
        n_samples: Number of training samples
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on
        viz_freq: Frequency to save visualizations (every N epochs)
        save_dir: Directory to save visualizations
        dim: Dimension ('1d' or '2d')
        cfm_type: CFM coupling type ('icfm', 'otcfm', 'uotcfm') - uotrfm NOT supported
        cfm_reg: Entropic regularization for Sinkhorn
        cfm_reg_m: Marginal regularization for unbalanced OT
        cfm_weight_power: Power factor for UOTRFM weights (unused, modefm does not support uotrfm)
        dataset_2d: 2D dataset type
        lr_scheduler: Learning rate scheduler type
        lr_scheduler_params: Scheduler parameters
        initial_sigma: Initial sigma for Gaussian kernel (default: 5.0 for 1D, 10.0 for 2D)
        min_sigma: Minimum sigma for annealing
        sigma_decay_factor: Sigma decay factor for 'exponential' schedule (default: 0.95)
        sigma_schedule: schedule name including 'batch_adaptive', 'sample_adaptive'
        sigma_schedule_params: dict with schedule-specific params
        use_var_head: add variance head for sample-adaptive sigma
        var_loss_weight: weight for variance head loss
        sigma_scale: kernel_width = scale * 2 * sigma^2 for sample_adaptive
        sample_adaptive_warmup_epochs: use fixed sigma for first N epochs (default 0)
        sample_adaptive_warmup_sigma: fixed sigma during warmup (default 10.0)

    Returns:
        model: Trained ModeFlowMatching model
        history: Training history (losses)
    """
    # UOTRFM is not supported
    if cfm_type == 'uotrfm':
        raise ValueError(
            "ModeFM does not support UOTRFM coupling. "
            "Use icfm, otcfm, or uotcfm. Please set cfm_type to one of these."
        )

    # sample_adaptive requires use_var_head
    use_sample_adaptive = (sigma_schedule == 'sample_adaptive')
    if use_sample_adaptive and not use_var_head:
        raise ValueError(
            "sample_adaptive sigma schedule requires use_var_head=True. "
            "Please set --var_head true."
        )

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sigma defaults: 1D=5.0, 2D=10.0 (different data scale)
    if initial_sigma is None:
        initial_sigma = 5.0 if dim == '1d' else 10.0

    # Build sigma schedule params (decay_factor for exponential)
    if sigma_schedule_params is None:
        sigma_schedule_params = {}
    sigma_schedule_params = dict(sigma_schedule_params)
    if 'decay_factor' not in sigma_schedule_params and sigma_schedule == 'exponential':
        sigma_schedule_params['decay_factor'] = sigma_decay_factor

    print("=" * 60)
    print("Training ModeFlowMatching (ModeFM) Model")
    print(f"CFM Type: {cfm_type.upper()}")
    print(f"Gaussian Kernel: sigma_schedule={sigma_schedule}, init={initial_sigma}, min={min_sigma}")
    if use_var_head:
        print(f"Var head: enabled, var_loss_weight={var_loss_weight}, sigma_scale={sigma_scale}")
    if use_sample_adaptive and sample_adaptive_warmup_epochs > 0:
        print(f"Sample_adaptive warmup: {sample_adaptive_warmup_epochs} epochs at sigma={sample_adaptive_warmup_sigma}")
    print("=" * 60)

    # Create CFM sampler
    cfm_sampler = create_cfm_sampler(
        cfm_type=cfm_type,
        reg=cfm_reg,
        reg_m=cfm_reg_m,
        weight_power=cfm_weight_power
    )

    # Determine input dimension
    input_dim = 1 if dim == '1d' else 2

    # Generate data
    print(f"Generating {n_samples} synthetic data samples ({dim})...")
    if dim == '1d':
        x_data = generate_data(n_samples=n_samples, seed=seed)
        x_data = torch.FloatTensor(x_data).unsqueeze(1).to(device)
    else:
        x_data = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
        x_data = torch.FloatTensor(x_data).to(device)

    # Create model
    model = ModeFlowMatching(
        input_dim=input_dim, initial_sigma=initial_sigma, use_var_head=use_var_head
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    if lr_scheduler_params is None:
        lr_scheduler_params = {}

    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01, **lr_scheduler_params
        )
    elif lr_scheduler == 'step':
        step_size = lr_scheduler_params.get('step_size', epochs // 3)
        gamma = lr_scheduler_params.get('gamma', 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler == 'exponential':
        gamma = lr_scheduler_params.get('gamma', 0.995)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler == 'none' or lr_scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {lr_scheduler}")

    # Training loop
    history = {'loss': []}
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Visualization imports
    if dim == '1d':
        from visualization.viz_modefm import visualize_modefm
    else:
        from visualization.viz_modefm_2d import visualize_modefm_2d

    os.makedirs(save_dir, exist_ok=True)

    z_train_viz = sample_prior(n_samples=n_samples, seed=seed, dim=input_dim)
    if dim == '1d':
        x_train_viz = generate_data(n_samples=n_samples, seed=seed)
    else:
        x_train_viz = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
    coupling_indices_viz = np.random.permutation(n_samples)

    n_infer = 200
    z_infer_viz = sample_prior(n_samples=n_infer, seed=seed + 1000, dim=input_dim)

    # Build adaptive sigma params for batch_adaptive schedule
    use_batch_adaptive = (sigma_schedule == 'batch_adaptive')
    if use_batch_adaptive:
        sigma_adaptive_params = {
            'gamma': sigma_schedule_params.get('gamma', 1.0),
            'q': sigma_schedule_params.get('q', 0.5),
            'eps': sigma_schedule_params.get('eps', 1e-6),
        }

    # Build sample_adaptive params (epoch and warmup updated per epoch in loop)
    if use_sample_adaptive:
        sample_adaptive_params = {
            'scale': sigma_scale,
            'eps': sigma_schedule_params.get('eps', 1e-8),
            'warmup_epochs': sample_adaptive_warmup_epochs,
            'warmup_sigma': sample_adaptive_warmup_sigma,
        }

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        # Sigma annealing (skip for batch_adaptive and sample_adaptive)
        if not use_batch_adaptive and not use_sample_adaptive:
            current_sigma = _compute_sigma(
                epoch, epochs, initial_sigma, min_sigma,
                sigma_schedule, sigma_schedule_params
            )
            model.update_sigma(current_sigma)

        model.train()
        epoch_loss = 0.0
        epoch_sigma_sum = 0.0
        epoch_sigma_count = 0

        # Shuffle data
        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_size_actual = end_idx - start_idx

            x_batch = x_shuffled[start_idx:end_idx]
            z_batch = torch.randn(batch_size_actual, input_dim).to(device)
            t_batch = torch.rand(batch_size_actual, 1).to(device)

            # Apply CFM coupling (weights will be None for icfm/otcfm/uotcfm in typical config)
            z_coupled, x_coupled, weights = cfm_sampler.sample_coupling(z_batch, x_batch)

            optimizer.zero_grad()
            if use_sample_adaptive:
                sample_adaptive_params['epoch'] = epoch
                loss, sigma_used = model.loss_function(
                    z_coupled, x_coupled, t_batch, weights=weights,
                    sample_adaptive_params=sample_adaptive_params,
                    var_loss_weight=var_loss_weight
                )
                epoch_sigma_sum += sigma_used * batch_size_actual
                epoch_sigma_count += batch_size_actual
            elif use_batch_adaptive:
                loss, sigma_used = model.loss_function(
                    z_coupled, x_coupled, t_batch,
                    weights=weights, sigma_adaptive_params=sigma_adaptive_params
                )
                epoch_sigma_sum += sigma_used * batch_size_actual
                epoch_sigma_count += batch_size_actual
            else:
                loss, _ = model.loss_function(
                    z_coupled, x_coupled, t_batch, weights=weights,
                    var_loss_weight=var_loss_weight if use_var_head else 0.0
                )
                sigma_used = current_sigma
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual

        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)
        if (use_batch_adaptive or use_sample_adaptive) and epoch_sigma_count > 0:
            current_sigma = epoch_sigma_sum / epoch_sigma_count
        # For non-adaptive, current_sigma was set at start of epoch

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if use_sample_adaptive:
                warmup = (sample_adaptive_warmup_epochs > 0 and epoch < sample_adaptive_warmup_epochs)
                sigma_str = f"Sigma: {current_sigma:.4f} (sample_adaptive{' warmup' if warmup else ''})"
            elif use_batch_adaptive:
                sigma_str = f"Sigma: {current_sigma:.4f} (batch_adaptive)"
            else:
                sigma_str = f"Sigma: {current_sigma:.4f}"
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, {sigma_str}, LR: {current_lr:.6e}")

        # Visualization
        if (epoch + 1) % viz_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                if dim == '1d':
                    z_tensor = torch.FloatTensor(z_infer_viz).unsqueeze(1).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.squeeze().cpu().numpy()

                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_modefm(
                        z_samples=z_train_viz,
                        x_data=x_train_viz,
                        trajectories=trajectories,
                        coupling_indices=coupling_indices_viz,
                        save_path=viz_path,
                        vector_info=None,
                        mcc_sigma=current_sigma
                    )
                else:
                    z_tensor = torch.FloatTensor(z_infer_viz).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.cpu().numpy()

                    viz_path = os.path.join(save_dir, f'epoch_{epoch+1:04d}.png')
                    visualize_modefm_2d(
                        trajectories=trajectories,
                        x_data=x_train_viz,
                        save_path=viz_path,
                        epoch=epoch + 1,
                        cfm_type=cfm_type,
                        mcc_sigma=current_sigma
                    )
            model.train()

    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")

    return model, history
