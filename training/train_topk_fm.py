"""
Training script for TopK-OTCFM model
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.topk_fm import TopKFlowMatching
from data.synthetic import generate_data, generate_data_2d
from utils.cfm_sampler import create_cfm_sampler


def get_top_filter_k_schedule(epoch, retrain_start_epoch, total_retrain_epochs, 
                               schedule_type='fixed', k_start=1.0, k_end=0.1):
    """
    Compute top_filter_k value based on schedule
    
    Args:
        epoch: Current epoch (0-indexed)
        retrain_start_epoch: Epoch when retraining starts
        total_retrain_epochs: Total number of retraining epochs
        schedule_type: 'fixed', 'linear', 'exponential', 'cosine'
        k_start: Initial top_filter_k value (at retrain_start_epoch)
        k_end: Final top_filter_k value (at end of training)
    
    Returns:
        top_filter_k: Current value for top_filter_k
    """
    if epoch < retrain_start_epoch:
        return k_start  # Pretraining: use start value
    
    # Compute progress in retraining phase [0, 1]
    retrain_epoch = epoch - retrain_start_epoch
    progress = retrain_epoch / total_retrain_epochs
    progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
    
    if schedule_type == 'fixed':
        return k_start
    elif schedule_type == 'linear':
        return k_start - (k_start - k_end) * progress
    elif schedule_type == 'exponential':
        # Exponential decay: k = k_start * (k_end / k_start) ^ progress
        if k_start <= 0 or k_end <= 0:
            return k_start  # Fallback to fixed if invalid values
        return k_start * ((k_end / k_start) ** progress)
    elif schedule_type == 'cosine':
        # Cosine annealing: k = k_end + (k_start - k_end) * (1 + cos(π * progress)) / 2
        return k_end + (k_start - k_end) * (1 + math.cos(math.pi * progress)) / 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def train_topk_fm(n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
                  viz_freq=200, save_dir='/home/user/Desktop/Gen_Study/outputs', dim='1d',
                  cfm_type='otcfm', cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
                  topk_pretrain_epochs=150, top_filter_k=0.5, 
                  top_filter_k_schedule='fixed', top_filter_k_start=1.0, top_filter_k_end=0.1,
                  ode_solver='dopri5', ode_tol=1e-5, dataset_2d='2gauss',
                  lr_scheduler='cosine', lr_scheduler_params=None):
    """
    Train TopK-OTCFM model
    
    Training stages:
    1. Pretraining (epochs 0 to topk_pretrain_epochs): Standard OTCFM loss
    2. Retraining (epochs topk_pretrain_epochs+1 to epochs): OT coupling + ODE + TopK selection
    
    Args:
        n_samples: Number of training samples
        epochs: Total number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        seed: Random seed
        device: Device to train on
        viz_freq: Frequency to save visualizations (every N epochs)
        save_dir: Directory to save visualizations
        dim: Dimension ('1d' or '2d')
        cfm_type: CFM coupling type (should be 'otcfm' for TopK-OTCFM)
        cfm_reg: Entropic regularization for Sinkhorn
        cfm_reg_m: Marginal regularization for unbalanced OT
        cfm_weight_power: Power factor for UOTRFM weights
        topk_pretrain_epochs: Number of epochs for pretraining (default: 150)
        top_filter_k: Fixed value if schedule='fixed', or initial value if schedule != 'fixed' (default: 0.5)
        top_filter_k_schedule: Schedule type for top_filter_k ('fixed', 'linear', 'exponential', 'cosine', default: 'fixed')
        top_filter_k_start: Starting value for top_filter_k at retraining start (default: 1.0)
        top_filter_k_end: Ending value for top_filter_k at end of training (default: 0.1)
        ode_solver: ODE solver method ('dopri5' or 'euler', default: 'dopri5')
        ode_tol: Tolerance for adaptive ODE solver (default: 1e-5)
    
    Returns:
        model: Trained TopK-OTCFM model
        history: Training history (losses)
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    total_retrain_epochs = epochs - topk_pretrain_epochs
    
    print("=" * 60)
    print("Training TopK-OTCFM Model")
    print(f"CFM Type: {cfm_type.upper()}")
    print(f"Pretraining epochs: {topk_pretrain_epochs}")
    print(f"Retraining epochs: {total_retrain_epochs}")
    print(f"Top filter k schedule: {top_filter_k_schedule}")
    if top_filter_k_schedule == 'fixed':
        print(f"Top filter k (fixed): {top_filter_k}")
    else:
        print(f"Top filter k: {top_filter_k_start} -> {top_filter_k_end} ({top_filter_k_schedule})")
    print(f"ODE solver: {ode_solver}")
    print("=" * 60)
    
    # Create CFM sampler (use OTCFM for coupling)
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
    else:  # 2d
        x_data = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
        x_data = torch.FloatTensor(x_data).to(device)
    
    # Create model
    model = TopKFlowMatching(
        input_dim=input_dim,
        ode_solver=ode_solver,
        ode_tol=ode_tol
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    if lr_scheduler_params is None:
        lr_scheduler_params = {}
    
    if lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01, **lr_scheduler_params)
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
    history = {
        'loss': [],
        'loss_pretrain': [],
        'loss_retrain': [],
        'loss_topk': [],
        'loss_full': [],
        'k_selected': [],
        'nfe': []  # Average NFE per batch
    }
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Prepare visualization data
    if dim == '1d':
        from visualization.viz_topk_fm import visualize_topk_fm
    else:
        from visualization.viz_topk_fm_2d import visualize_topk_fm_2d
    
    from data.synthetic import sample_prior
    
    os.makedirs(save_dir, exist_ok=True)
    
    z_train_viz = sample_prior(n_samples=n_samples, seed=seed, dim=input_dim)
    if dim == '1d':
        x_train_viz = generate_data(n_samples=n_samples, seed=seed)
    else:
        x_train_viz = generate_data_2d(n_samples=n_samples, seed=seed, dataset=dataset_2d)
    coupling_indices_viz = np.random.permutation(n_samples)
    
    n_infer = 200
    z_infer_viz = sample_prior(n_samples=n_infer, seed=seed + 1000, dim=input_dim)
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_pretrain = 0.0
        epoch_loss_retrain = 0.0
        epoch_loss_topk = 0.0
        epoch_loss_full = 0.0
        epoch_k_selected = 0.0
        epoch_nfe_sum = 0.0  # Sum of NFE across batches
        
        # Determine training stage
        is_pretraining = epoch < topk_pretrain_epochs
        
        # Compute current top_filter_k value based on schedule
        if top_filter_k_schedule == 'fixed':
            current_top_filter_k = top_filter_k
        else:
            current_top_filter_k = get_top_filter_k_schedule(
                epoch=epoch,
                retrain_start_epoch=topk_pretrain_epochs,
                total_retrain_epochs=total_retrain_epochs,
                schedule_type=top_filter_k_schedule,
                k_start=top_filter_k_start,
                k_end=top_filter_k_end
            )
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        x_shuffled = x_data[indices]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_size_actual = end_idx - start_idx
            
            x_batch = x_shuffled[start_idx:end_idx]
            z_batch = torch.randn(batch_size_actual, input_dim).to(device)
            
            optimizer.zero_grad()
            
            # Sample time for FM loss
            t_batch = torch.rand(batch_size_actual, 1).to(device)
            
            if is_pretraining:
                # Pretraining: Standard OTCFM loss
                # Apply OT coupling
                z_coupled, x_coupled, weights = cfm_sampler.sample_coupling(z_batch, x_batch)
                
                loss = model.loss_function_pretrain(z_coupled, x_coupled, t_batch, weights=weights)
                epoch_loss_pretrain += loss.item() * batch_size_actual
            else:
                # Retraining: OT coupling + ODE + TopK selection
                # Apply OT coupling to get (x0, x1) pairs
                z_coupled, x_coupled, weights = cfm_sampler.sample_coupling(z_batch, x_batch)
                
                # Compute retraining loss with TopK selection
                # ODE error is used for sample selection, FM loss is used for training
                loss, logs = model.loss_function_retrain(
                    z_coupled, x_coupled, t_batch,
                    top_filter_k=current_top_filter_k,  # Use scheduled value
                    device=device,
                    weights=weights
                )
                
                epoch_loss_retrain += loss.item() * batch_size_actual
                epoch_loss_topk += logs['loss_fm'] * batch_size_actual
                epoch_loss_full += logs['mean_ode_error'] * batch_size_actual
                epoch_k_selected += logs['k_selected'] * batch_size_actual
                epoch_nfe_sum += logs['nfe']  # NFE is per batch, not per sample
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_size_actual
        
        # Average losses
        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)
        
        if is_pretraining:
            epoch_loss_pretrain /= n_samples
            history['loss_pretrain'].append(epoch_loss_pretrain)
            history['loss_retrain'].append(0.0)
            history['loss_topk'].append(0.0)
            history['loss_full'].append(0.0)
            history['k_selected'].append(0)
            history['nfe'].append(0.0)
        else:
            epoch_loss_retrain /= n_samples
            epoch_loss_topk /= n_samples
            epoch_loss_full /= n_samples
            epoch_k_selected /= n_batches  # Average k per batch
            epoch_nfe_avg = epoch_nfe_sum / n_batches  # Average NFE per batch
            
            history['loss_pretrain'].append(0.0)
            history['loss_retrain'].append(epoch_loss_retrain)
            history['loss_topk'].append(epoch_loss_topk)
            history['loss_full'].append(epoch_loss_full)
            history['k_selected'].append(int(epoch_k_selected))
            history['nfe'].append(epoch_nfe_avg)
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        if (epoch + 1) % 200 == 0 or epoch == 0:
            stage = "Pretraining" if is_pretraining else "Retraining"
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] ({stage}), Loss: {epoch_loss:.6f}, LR: {current_lr:.6e}")
            if not is_pretraining:
                print(f"  TopK loss: {epoch_loss_topk:.6f}, Full error: {epoch_loss_full:.6f}, "
                      f"k selected: {int(epoch_k_selected)}, top_filter_k: {current_top_filter_k:.4f}")
        
        # Generate visualization every viz_freq epochs
        if (epoch + 1) % viz_freq == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                if dim == '1d':
                    # Generate ODE trajectories
                    z_tensor = torch.FloatTensor(z_infer_viz).unsqueeze(1).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.squeeze().cpu().numpy()
                    
                    # Save visualization
                    viz_path = os.path.join(save_dir, f'topk_fm_epoch_{epoch+1:04d}.png')
                    visualize_topk_fm(
                        z_samples=z_train_viz,
                        x_data=x_train_viz,
                        trajectories=trajectories,
                        coupling_indices=coupling_indices_viz,
                        save_path=viz_path,
                        epoch=epoch + 1,
                        is_pretraining=is_pretraining
                    )
                else:  # 2d
                    # Generate trajectories
                    z_tensor = torch.FloatTensor(z_infer_viz).to(device)
                    trajectories_tensor = model.sample(z_tensor, n_steps=100, device=device)
                    trajectories = trajectories_tensor.cpu().numpy()  # (n_steps+1, n_infer, 2)
                    
                    # Save visualization
                    viz_path = os.path.join(save_dir, f'topk_fm_epoch_{epoch+1:04d}.png')
                    # Get current epoch's NFE (0.0 for pretraining)
                    current_nfe = history['nfe'][-1] if len(history['nfe']) > 0 else 0.0
                    visualize_topk_fm_2d(
                        trajectories=trajectories,
                        x_data=x_train_viz,
                        save_path=viz_path,
                        epoch=epoch + 1,
                        is_pretraining=is_pretraining,
                        cfm_type=cfm_type,
                        nfe=current_nfe if not is_pretraining else None
                    )
            model.train()
    
    print("\nTraining complete!")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TopK-OTCFM model')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Gen_Study/outputs',
                        help='Directory to save model and visualization')
    parser.add_argument('--viz_freq', type=int, default=200,
                        help='Frequency to save visualizations (every N epochs)')
    parser.add_argument('--topk_pretrain_epochs', type=int, default=150,
                        help='Number of epochs for pretraining')
    parser.add_argument('--top_filter_k', type=float, default=0.5,
                        help='Fixed top_filter_k value (if schedule=fixed) or initial value (0 < k <= 1)')
    parser.add_argument('--top_filter_k_schedule', type=str, default='fixed',
                        choices=['fixed', 'linear', 'exponential', 'cosine'],
                        help='Schedule type for top_filter_k (default: fixed)')
    parser.add_argument('--top_filter_k_start', type=float, default=1.0,
                        help='Starting value for top_filter_k at retraining start (default: 1.0)')
    parser.add_argument('--top_filter_k_end', type=float, default=0.1,
                        help='Ending value for top_filter_k at end of training (default: 0.1)')
    parser.add_argument('--ode_solver', type=str, default='dopri5', choices=['dopri5', 'euler'],
                        help='ODE solver method')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train model
    model, history = train_topk_fm(
        n_samples=args.n_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        viz_freq=args.viz_freq,
        save_dir=args.output_dir,
        topk_pretrain_epochs=args.topk_pretrain_epochs,
        top_filter_k=args.top_filter_k,
        top_filter_k_schedule=args.top_filter_k_schedule,
        top_filter_k_start=args.top_filter_k_start,
        top_filter_k_end=args.top_filter_k_end,
        ode_solver=args.ode_solver
    )
    
    save_path = os.path.join(args.output_dir, 'topk_fm_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        z_test = torch.randn(10, 1).to(device)
        trajectory = model.sample(z_test, n_steps=100, device=device)
        print(f"Sample trajectory shape: {trajectory.shape}")
        print(f"Final samples: {trajectory[-1].squeeze().cpu().numpy()}")
