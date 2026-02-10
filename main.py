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
from models.improved_mean_flow import ImprovedMeanFlow
from models.tdmf import TDMF
from models.topk_fm import TopKFlowMatching
from models.novae import NOVAE
from training.train_decoder import train_decoder
from training.train_ae import train_autoencoder
from training.train_vae import train_vae
from training.train_fm import train_fm
from training.train_meanflow import train_meanflow
from training.train_imf import train_imf
from training.train_tdmf import train_tdmf
from training.train_facm import train_facm
from training.train_backflow import train_backflow
from training.train_topk_fm import train_topk_fm
from training.train_novae import train_novae
from visualization.viz_decoder import visualize_decoder
from visualization.viz_ae import visualize_autoencoder
from visualization.viz_vae import visualize_vae
from visualization.viz_fm import visualize_fm
from visualization.viz_meanflow import visualize_meanflow
from visualization.viz_facm import visualize_facm
from visualization.viz_imf import visualize_imf
from visualization.viz_backflow import (
    visualize_backflow,
    visualize_training_coupling,
    euler_solve,
    one_step_decode,
    compute_trajectories
)
from visualization.viz_topk_fm import visualize_topk_fm
from visualization.viz_novae import visualize_novae


def run_experiment(model_type, n_samples=500, epochs=2000, lr=1e-3, batch_size=64, seed=42, device='cpu',
                   cfm_type='icfm', cfm_reg=0.05, cfm_reg_m=(float('inf'), 2.0), cfm_weight_power=10.0,
                   lambda_trans=0.1, lambda_schedule='fixed',
                   topk_pretrain_epochs=150, top_filter_k=0.5, ode_solver='dopri5', ode_tol=1e-5):
    """
    Args:
        model_type (str): Type of model to run ('decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'imf', 'facm', 'backflow', 'novae').
        cfm_type (str): CFM coupling type ('icfm', 'otcfm', 'uotcfm', 'uotrfm')
        cfm_reg (float): Entropic regularization for Sinkhorn
        cfm_reg_m (tuple): Marginal regularization for unbalanced OT
        cfm_weight_power (float): Power factor for UOTRFM weights
        ... other args ...
    """
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {model_type.upper()}")
    if model_type in ['fm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm']:
        print(f"CFM Type: {cfm_type.upper()}")
    if model_type == 'tdmf':
        print(f"Lambda Trans: {lambda_trans}, Schedule: {lambda_schedule}")
    if model_type == 'topk_fm':
        print(f"Pretrain Epochs: {topk_pretrain_epochs}, Top Filter K: {top_filter_k}, ODE Solver: {ode_solver}")
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
            model, _ = train_fm(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device,
                               cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power)
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
            model, _ = train_meanflow(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device,
                                     cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power)
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

    elif model_type == 'imf':
        # ========================================
        # 7. Improved MeanFlow (iMF)
        # ========================================
        imf_model_path = '/home/user/Desktop/Gen_Study/outputs/imf_model.pt'
        if os.path.exists(imf_model_path):
            print(f"Loading existing model from {imf_model_path}")
            model = ImprovedMeanFlow(input_dim=1).to(device)
            model.load_state_dict(torch.load(imf_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_imf(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device,
                                cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power)
            torch.save(model.state_dict(), imf_model_path)

        model.eval()
        print("\nPreparing Improved MeanFlow visualization...")

        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)

        with torch.no_grad():
            # ODE trajectories
            trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
            trajectories = trajectories_tensor.squeeze().cpu().numpy()

            # Mean velocity ODE trajectories (u multi-step, 2 steps)
            mean_trajectories_tensor = model.sample_mean_velocity_ode(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=2, device=device)
            mean_trajectories = mean_trajectories_tensor.squeeze().cpu().numpy()

            # One-step predictions using mean velocity u
            mean_predictions_tensor = model.sample_mean_velocity(torch.FloatTensor(z_infer).unsqueeze(1).to(device), device=device)
            mean_predictions = mean_predictions_tensor.squeeze().cpu().numpy()

        visualize_imf(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            mean_predictions=mean_predictions,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/imf_visualization.png',
            mean_trajectories=mean_trajectories
        )

    elif model_type == 'tdmf':
        # ========================================
        # 8. Translation Decoupled MeanFlow (TDMF)
        # ========================================
        tdmf_model_path = '/home/user/Desktop/Gen_Study/outputs/tdmf_model.pt'
        if os.path.exists(tdmf_model_path):
            print(f"Loading existing model from {tdmf_model_path}")
            model = TDMF(input_dim=1, lambda_trans=lambda_trans, lambda_schedule=lambda_schedule).to(device)
            model.load_state_dict(torch.load(tdmf_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_tdmf(n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, device=device,
                                 cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power,
                                 lambda_trans=lambda_trans, lambda_schedule=lambda_schedule)
            torch.save(model.state_dict(), tdmf_model_path)

        model.eval()
        print("\nPreparing Translation Decoupled MeanFlow visualization...")

        # Training data for context
        z_train = sample_prior(n_samples=n_samples, seed=seed)
        x_train = generate_data(n_samples=n_samples, seed=seed)
        coupling_indices = np.random.permutation(n_samples)

        # Inference data
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 1)

        with torch.no_grad():
            # ODE trajectories
            trajectories_tensor = model.sample(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=100, device=device)
            trajectories = trajectories_tensor.squeeze().cpu().numpy()

            # Mean velocity ODE trajectories (u multi-step, 2 steps)
            mean_trajectories_tensor = model.sample_mean_velocity_ode(torch.FloatTensor(z_infer).unsqueeze(1).to(device), n_steps=2, device=device)
            mean_trajectories = mean_trajectories_tensor.squeeze().cpu().numpy()

            # One-step predictions using mean velocity u
            mean_predictions_tensor = model.sample_mean_velocity(torch.FloatTensor(z_infer).unsqueeze(1).to(device), device=device)
            mean_predictions = mean_predictions_tensor.squeeze().cpu().numpy()

        visualize_imf(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            mean_predictions=mean_predictions,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/tdmf_visualization.png',
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
                dim='1d',
                cfm_type=cfm_type,
                cfm_reg=cfm_reg,
                cfm_reg_m=cfm_reg_m,
                cfm_weight_power=cfm_weight_power
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
                save_dir='/home/user/Desktop/Gen_Study/outputs',
                cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power
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

    elif model_type == 'topk_fm':
        # ========================================
        # 9. TopK-OTCFM
        # ========================================
        topk_fm_model_path = '/home/user/Desktop/Gen_Study/outputs/topk_fm_model.pt'
        if os.path.exists(topk_fm_model_path):
            print(f"Loading existing model from {topk_fm_model_path}")
            model = TopKFlowMatching(input_dim=1, ode_solver=ode_solver, ode_tol=ode_tol).to(device)
            model.load_state_dict(torch.load(topk_fm_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_topk_fm(
                n_samples=n_samples, epochs=epochs, lr=lr, batch_size=batch_size,
                seed=seed, device=device, viz_freq=200,
                save_dir='/home/user/Desktop/Gen_Study/outputs',
                cfm_type=cfm_type, cfm_reg=cfm_reg, cfm_reg_m=cfm_reg_m, cfm_weight_power=cfm_weight_power,
                topk_pretrain_epochs=topk_pretrain_epochs, top_filter_k=top_filter_k,
                ode_solver=ode_solver, ode_tol=ode_tol
            )
            torch.save(model.state_dict(), topk_fm_model_path)

        model.eval()
        print("\nPreparing TopK-OTCFM visualization...")

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

        visualize_topk_fm(
            z_samples=z_train,
            x_data=x_train,
            trajectories=trajectories,
            coupling_indices=coupling_indices,
            save_path='/home/user/Desktop/Gen_Study/outputs/topk_fm_visualization.png',
            epoch=epochs,
            is_pretraining=False
        )

    elif model_type == 'novae':
        # ========================================
        # 10. NO-VAE (Noise Oriented VAE)
        # ========================================
        novae_model_path = '/home/user/Desktop/Gen_Study/outputs/novae_model.pt'
        if os.path.exists(novae_model_path):
            print(f"Loading existing model from {novae_model_path}")
            model = NOVAE(input_dim=1, latent_dim=1).to(device)
            model.load_state_dict(torch.load(novae_model_path, map_location=device))
        else:
            print("Training new model...")
            model, _ = train_novae(
                n_samples=n_samples, epochs=epochs, lr=lr,
                batch_size=batch_size, seed=seed, device=device,
                coupling_method='sinkhorn',
                sinkhorn_reg=0.05,
                sinkhorn_reg_schedule='cosine',
                sinkhorn_reg_init=1.0,
                sinkhorn_reg_final=0.01,
                z_recon_weight=1.0
            )
            torch.save(model.state_dict(), novae_model_path)

        model.eval()
        print("\nPreparing NO-VAE visualization...")
        x_data = generate_data(n_samples=n_samples, seed=seed)

        # Encode training data to see encoder output distribution
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_data).unsqueeze(1).to(device)
            z_ = model.encode(x_tensor).squeeze().cpu().numpy()

            # Reconstruct through full pipeline (encode -> soft NN -> decode)
            z_prior_viz = torch.randn(model.n_prior_samples, 1).to(device)
            x_hat_out, _, z_sel, _ = model(x_tensor, z_prior_viz)
            x_hat = x_hat_out.squeeze().cpu().numpy()

        # Inference: sample from prior and decode
        n_infer = 200
        z_infer = sample_prior(n_samples=n_infer, seed=seed + 5)
        with torch.no_grad():
            x_infer = model.decode(
                torch.FloatTensor(z_infer).unsqueeze(1).to(device)
            ).squeeze().cpu().numpy()

        visualize_novae(
            z_prior=sample_prior(n_samples, seed=seed + 4),
            x_data=x_data,
            z_=z_,
            x_hat=x_hat,
            z_infer=z_infer,
            x_infer=x_infer,
            save_path='/home/user/Desktop/Gen_Study/outputs/novae_visualization.png',
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
        choices=['all', 'decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm', 'novae'],
        help='Specify which model to run.'
    )
    parser.add_argument(
        '--cfm',
        type=str,
        default='icfm',
        choices=['icfm', 'otcfm', 'uotcfm', 'uotrfm'],
        help='CFM coupling type for flow models (default: icfm)'
    )
    parser.add_argument(
        '--cfm_weight_power',
        type=float,
        default=10.0,
        help='Power factor for UOTRFM weights (default: 10.0)'
    )
    parser.add_argument(
        '--lambda_trans',
        type=float,
        default=0.1,
        help='Weight for translation loss in TDMF (default: 0.1)'
    )
    parser.add_argument(
        '--lambda_schedule',
        type=str,
        default='fixed',
        choices=['fixed', 'linear'],
        help='Lambda schedule type for TDMF (default: fixed)'
    )
    parser.add_argument(
        '--topk_pretrain_epochs',
        type=int,
        default=150,
        help='Number of epochs for TopK-OTCFM pretraining (default: 150)'
    )
    parser.add_argument(
        '--top_filter_k',
        type=float,
        default=0.5,
        help='Fraction of samples to update in TopK-OTCFM retraining (0 < k <= 1, default: 0.5)'
    )
    parser.add_argument(
        '--ode_solver',
        type=str,
        default='dopri5',
        choices=['dopri5', 'euler'],
        help='ODE solver method for TopK-OTCFM (default: dopri5)'
    )
    parser.add_argument(
        '--ode_tol',
        type=float,
        default=1e-5,
        help='Tolerance for adaptive ODE solver (default: 1e-5)'
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
        "device": device,
        "cfm_type": args.cfm,
        "cfm_reg": 0.05,
        "cfm_reg_m": (float('inf'), 2.0),
        "cfm_weight_power": args.cfm_weight_power,
        "lambda_trans": args.lambda_trans,
        "lambda_schedule": args.lambda_schedule,
        "topk_pretrain_epochs": args.topk_pretrain_epochs,
        "top_filter_k": args.top_filter_k,
        "ode_solver": args.ode_solver,
        "ode_tol": args.ode_tol
    }

    print("\n" + "=" * 80)
    print("GENERATIVE MODELS STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, val in params.items():
        if key == 'cfm_reg_m':
            print(f"  - {key}: {val}")
        else:
            print(f"  - {key}: {val}")
    print("\n" + "=" * 80)

    if args.model == 'all':
        all_models = ['decoder', 'ae', 'vae', 'vae_beta', 'fm', 'meanflow', 'imf', 'tdmf', 'facm', 'backflow', 'topk_fm', 'novae']
        for model_type in all_models:
            run_experiment(model_type, **params)
    else:
        run_experiment(args.model, **params)

    print("\n\n" + "=" * 80)
    print("ALL REQUESTED EXPERIMENTS COMPLETE!")
    print("=" * 80)
    print("\nCheck the 'outputs' directory for models and visualizations.")
    print("\n" + "=" * 80 + "\n")
