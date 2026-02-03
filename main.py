"""
Main execution script for all generative model experiments
"""
import torch
import numpy as np
import os
import sys
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic import generate_data, sample_prior
from models.decoder import Decoder
from models.autoencoder import Autoencoder
from models.vae import VAE
from models.flow_matching import FlowMatching
from models.mean_flow import MeanFlow
from models.facm import FACM
from models.backflow import BackFlow
from training.train_decoder import train_decoder
from training.train_ae import train_autoencoder
from training.train_vae import train_vae
from training.train_fm import train_fm
from training.train_meanflow import train_meanflow
from training.train_facm import train_facm
from training.train_backflow import train_backflow
from visualization.viz_decoder import visualize_decoder
from visualization.viz_ae import visualize_autoencoder
from visualization.viz_vae import visualize_vae
from visualization.viz_fm import visualize_fm
from visualization.viz_meanflow import visualize_meanflow
from visualization.viz_facm import visualize_facm
from visualization.viz_backflow import (
    visualize_backflow,
    visualize_training_coupling,
    euler_solve,
    one_step_decode,
    compute_trajectories
)


def run_experiment(model_type, n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu'):
    """
    Args:
        model_type (str): Type of model to run ('decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'facm', 'backflow').
        ... other args ...
    """
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {model_type.upper()}")
    print("=" * 80)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create outputs directory
    os.makedirs('/home/user/Desktop/Gen_Study/outputs', exist_ok=True)

    if model_type == 'decoder':
        # ========================================
        # 1. Non-identifiable Decoder
        # ========================================
        decoder_model_path = '/home/user/Desktop/Gen_Study/outputs/nid_decoder_model.pt'
        if os.path.exists(decoder_model_path):
            print(f"Loading existing model from {decoder_model_path}")
            model = Decoder(latent_dim=1, output_dim=1).to(device)
            model.load_state_dict(torch.load(decoder_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_decoder(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)
            torch.save(model.state_dict(), decoder_model_path)
        
        model.eval()
        print("\nPreparing decoder visualization...")
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)
        with torch.no_grad():
            x_hat = model(torch.FloatTensor(z_infer).unsqueeze(1).to(device)).squeeze().cpu().numpy()

        visualize_decoder(
            z_samples=z_train, x_data=x_train, x_hat=x_hat, z_infer=z_infer,
            coupling_indices=coupling_indices, save_path='/home/user/Desktop/Gen_Study/outputs/nid_decoder_visualization.png'
        )

    elif model_type == 'ae':
        # ========================================
        # 2. Autoencoder
        # ========================================
        ae_model_path = '/home/user/Desktop/Gen_Study/outputs/autoencoder_model.pt'
        if os.path.exists(ae_model_path):
            print(f"Loading existing model from {ae_model_path}")
            model = Autoencoder(input_dim=1, latent_dim=1).to(device)
            model.load_state_dict(torch.load(ae_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_autoencoder(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)
            torch.save(model.state_dict(), ae_model_path)
            
        model.eval()
        print("\nPreparing autoencoder visualization...")
        x_data = generate_data(n_samples=n_samples, seed=seed)
        with torch.no_grad():
            x_hat, z_hat = model(torch.FloatTensor(x_data).unsqueeze(1).to(device))
            x_hat, z_hat = x_hat.squeeze().cpu().numpy(), z_hat.squeeze().cpu().numpy()
        
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 3)
        with torch.no_grad():
            x_infer = model.decode(torch.FloatTensor(z_infer).unsqueeze(1).to(device)).squeeze().cpu().numpy()

        visualize_autoencoder(
            z_prior=sample_prior(n_samples, seed=seed + 2), x_data=x_data, z_hat=z_hat, x_hat=x_hat,
            z_infer=z_infer, x_infer=x_infer, save_path='/home/user/Desktop/Gen_Study/outputs/autoencoder_visualization.png'
        )

    elif 'vae' in model_type:
        # ========================================
        # 3. & 4. VAEs
        # ========================================
        beta = 10.0 if model_type == 'vae_beta' else 1.0
        vae_model_path = f'/home/user/Desktop/Gen_Study/outputs/vae_beta{beta}_model.pt'
        
        if os.path.exists(vae_model_path):
            print(f"Loading existing model from {vae_model_path}")
            model = VAE(input_dim=1, latent_dim=1, beta=beta).to(device)
            model.load_state_dict(torch.load(vae_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_vae(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device, beta=beta)
            torch.save(model.state_dict(), vae_model_path)
            
        model.eval()
        print(f"\nPreparing VAE (beta={beta}) visualization...")
        x_data = generate_data(n_samples=n_samples, seed=seed)
        with torch.no_grad():
            x_hat, _, _, z_hat = model(torch.FloatTensor(x_data).unsqueeze(1).to(device))
            x_hat, z_hat = x_hat.squeeze().cpu().numpy(), z_hat.squeeze().cpu().numpy()

        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 4)
        with torch.no_grad():
            x_infer = model.decode(torch.FloatTensor(z_infer).unsqueeze(1).to(device)).squeeze().cpu().numpy()

        visualize_vae(
            z_prior=sample_prior(n_samples, seed=seed + 3), x_data=x_data, z_hat=z_hat, x_hat=x_hat,
            z_infer=z_infer, x_infer=x_infer, save_path=f'/home/user/Desktop/Gen_Study/outputs/vae_beta{beta}_visualization.png', beta=beta
        )

    elif model_type == 'fm':
        # ========================================
        # 5. Flow Matching
        # ========================================
        fm_model_path = '/home/user/Desktop/Gen_Study/outputs/fm_model.pt'
        if os.path.exists(fm_model_path):
            print(f"Loading existing model from {fm_model_path}")
            model = FlowMatching(input_dim=1).to(device)
            model.load_state_dict(torch.load(fm_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_fm(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)
            torch.save(model.state_dict(), fm_model_path)
            
        model.eval()
        print("\nPreparing Flow Matching visualization...")
        
        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)
        with torch.no_grad():
            trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
            trajectories = trajectories_tensor.squeeze().cpu().numpy()

        visualize_fm(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/fm_visualization.png'
        )

    elif model_type == 'meanflow':
        # ========================================
        # 6. MeanFlow
        # ========================================
        meanflow_model_path = '/home/user/Desktop/Gen_Study/outputs/meanflow_model.pt'
        if os.path.exists(meanflow_model_path):
            print(f"Loading existing model from {meanflow_model_path}")
            model = MeanFlow(input_dim=1).to(device)
            model.load_state_dict(torch.load(meanflow_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_meanflow(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device)
            torch.save(model.state_dict(), meanflow_model_path)

        model.eval()
        print("\nPreparing MeanFlow visualization...")

        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)

        with torch.no_grad():
            # ODE trajectories using instantaneous velocity v
            trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
            trajectories = trajectories_tensor.squeeze().cpu().numpy()

            # Mean velocity ODE trajectories (u multi-step, 2 steps)
            mean_trajectories_tensor = model.sample_mean_velocity_ode(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=2, device=device)
            mean_trajectories = mean_trajectories_tensor.squeeze().cpu().numpy()

            # One-step predictions using mean velocity u
            mean_predictions_tensor = model.sample_mean_velocity(torch.FloatTensor(z_infer).unsqueeze(1).to(device), device=device)
            mean_predictions = mean_predictions_tensor.squeeze().cpu().numpy()

        visualize_meanflow(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            mean_predictions=mean_predictions,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/meanflow_visualization.png',
            mean_trajectories=mean_trajectories
        )

    elif model_type == 'facm':
        # ========================================
        # 7. FACM (Flow-Anchored Consistency Models)
        # ========================================
        facm_model_path = '/home/user/Desktop/Gen_Study/outputs/facm_model.pt'
        if os.path.exists(facm_model_path):
            print(f"Loading existing model from {facm_model_path}")
            model = FACM(input_dim=1).to(device)
            model.load_state_dict(torch.load(facm_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_facm(
                n_samples=n_samples,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                seed=seed,
                device=device,
                dim='1d'
            )
            torch.save(model.state_dict(), facm_model_path)

        model.eval()
        print("\nPreparing FACM visualization...")

        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)
        z_tensor = torch.FloatTensor(z_infer).unsqueeze(1).to(device)

        with torch.no_grad():
            trajectories_tensor = model.sample_euler(z_tensor, n_steps=100, heun=False, timestep_shift=0.0)
            trajectories = trajectories_tensor.squeeze().cpu().numpy()

            cm_onestep_tensor = model.sample_consistency(z_tensor, n_steps=1)
            cm_onestep = cm_onestep_tensor.squeeze().cpu().numpy()

        visualize_facm(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            cm_onestep=cm_onestep,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/facm_visualization.png'
        )

    elif model_type == 'backflow':
        # ========================================
        # 8. BackFlow
        # ========================================
        backflow_model_path = '/home/user/Desktop/Gen_Study/outputs/backflow_model.pt'
        if os.path.exists(backflow_model_path):
            print(f"Loading existing model from {backflow_model_path}")
            model = BackFlow(input_dim=1).to(device)
            model.load_state_dict(torch.load(backflow_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_backflow(
                n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size,
                seed=seed, device=device, viz_freq=200,
                save_dir='/home/user/Desktop/Gen_Study/outputs'
            )
            torch.save(model.state_dict(), backflow_model_path)

        model.eval()
        print("\nPreparing BackFlow visualization...")

        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # 1. Save training coupling (once)
        visualize_training_coupling(
            z_samples=z_train,
            x_data=x_train,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/backflow_training_coupling.png'
        )

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)
        z_infer_tensor = torch.FloatTensor(z_infer).unsqueeze(1).to(device)

        with torch.no_grad():
            # Compute ODE trajectories using Euler method
            trajectories = compute_trajectories(model, z_infer_tensor, n_steps=100)

            # One-step decode
            onestep_final = one_step_decode(model, z_infer_tensor).squeeze().cpu().numpy()

        # 2. Save trajectory visualization (Euler + one-step)
        visualize_backflow(
            trajectories=trajectories,
            onestep_final=onestep_final,
            save_path='/home/user/Desktop/Gen_Study/outputs/backflow_visualization.png',
            x_data=x_train
        )

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {model_type.upper()} COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generative Model Experiments")
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'facm', 'backflow'],
        help='Specify which model to run.'
    )
    args = parser.parse_args()

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Common parameters
    params = {
        "n_samples": 500,
        "epochs": 2000,
        "lr": 1e-3,
        "batch_size": 64,
        "seed": 42,
        "device": device
    }

    print("\n" + "=" * 80)
    print("GENERATIVE MODELS STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, val in params.items():
        print(f"  - {key}: {val}")
    print("\n" + "=" * 80)

    if args.model == 'all':
        all_models = ['decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'facm', 'backflow']
        for model_type in all_models:
            run_experiment(model_type, **params)
    else:
        run_experiment(args.model, **params)

    print("\n\n" + "=" * 80)
    print("ALL REQUESTED EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print("\nCheck the 'outputs' directory for models and visualizations.")
    print("\n" + "=" * 80 + "\n")
