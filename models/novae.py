"""
Noise Oriented VAE (NO-VAE) model

Key idea: Replace VAE's Gaussian reparameterization with coupling between
encoded z_ and prior samples. The encoder outputs a single deterministic point
z_ instead of (mu, logvar). Among N prior samples, one is selected via coupling
(Sinkhorn OT or Soft Nearest Neighbor). The selected sample is then decoded
for reconstruction.

This naturally aligns the encoder output distribution with the prior without
explicit KL divergence or any regularization loss.

Coupling methods:
- 'sinkhorn': Optimal Transport coupling via Sinkhorn algorithm (ensures each
  noise sample is matched at most once)
- 'softnn': Soft Nearest Neighbor selection using softmax on negative squared
  L2 distances (allows multiple z_enc to select the same prior sample)
- 'ot_guided_soft': OT-Guided Soft Matching - combines 1:1 matching (Sinkhorn Hard)
  with soft alignment loss (Cross Entropy) to maximize probability of matched prior
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ot as pot
from models.base_mlp import BaseMLP


class NOVAE(nn.Module):
    """
    Noise Oriented VAE (NO-VAE)

    Training:
        Input: x (data sample), z_prior (N samples from prior)
        Encoding: x -> z_ (deterministic encoding, single point)
        Selection: z_ -> z_selected (OT coupling via Sinkhorn from z_prior)
        Decoding: z_selected -> x_hat
        Loss: MSE(x, x_hat) (reconstruction only, no regularization)

    Inference (Generation):
        Sample z ~ N(0, I) -> Decode -> x_hat (one-step, no ODE)

    The decoder always receives input that is a prior sample (via OT coupling),
    so it naturally learns to map from the prior distribution to the data distribution.
    OT coupling ensures each noise sample is matched at most once.

    Args:
        input_dim: Dimension of input/output space (default: 1)
        latent_dim: Dimension of latent space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
        n_prior_samples: Number of prior samples per mini-batch (default: None, uses batch_size or n_samples depending on context)
        coupling_method: Coupling method ('sinkhorn', 'softnn', or 'ot_guided_soft', default: 'sinkhorn')
        sinkhorn_reg: Entropic regularization for Sinkhorn (default: 0.05)
        normalize_cost: Whether to normalize cost matrix (default: True)
        sinkhorn_use_soft_coupling: For Sinkhorn, whether to use soft coupling (default: False)
        temperature: Temperature for Soft NN (default: 0.1)
        use_ste: For Soft NN, whether to use Straight-Through Estimator (default: False)
    """

    def __init__(self, input_dim=1, latent_dim=1, hidden_dims=None,
                 n_prior_samples=None, coupling_method='sinkhorn',
                 sinkhorn_reg=0.05, normalize_cost=True, sinkhorn_use_soft_coupling=False,
                 temperature=0.1, use_ste=False):
        super(NOVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_prior_samples = n_prior_samples
        self.coupling_method = coupling_method
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_reg_init = sinkhorn_reg  # Store initial value for annealing
        self.normalize_cost = normalize_cost
        # If None, auto-select based on sinkhorn_reg (soft if reg > 0.01)
        self.sinkhorn_use_soft_coupling = sinkhorn_use_soft_coupling
        # Soft NN parameters
        self.temperature = temperature
        self.temperature_init = temperature  # Store initial value for annealing
        self.use_ste = use_ste

        # Encoder: deterministic mapping x -> z_
        # Use reversed hidden dims for encoder (symmetric architecture)
        encoder_hidden = list(reversed(hidden_dims))
        self.encoder = BaseMLP(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=encoder_hidden
        )

        # Decoder: z -> x_hat
        self.decoder = BaseMLP(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=hidden_dims
        )

    def encode(self, x):
        """
        Deterministic encoding: x -> z_

        Args:
            x: Input of shape (batch_size, input_dim)

        Returns:
            z_: Encoder output of shape (batch_size, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent code to output space: z -> x_hat

        Args:
            z: Latent code of shape (batch_size, latent_dim)

        Returns:
            x_hat: Reconstructed output of shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def set_sinkhorn_reg(self, reg):
        """
        Update Sinkhorn regularization for annealing.
        
        Higher reg -> softer coupling (more spread out OT plan)
        Lower reg -> harder coupling (more concentrated OT plan)
        
        Args:
            reg: New regularization value (should decrease during training)
        """
        self.sinkhorn_reg = reg
    
    def set_temperature(self, temp):
        """
        Update temperature for Soft NN annealing.
        
        Higher temp -> softer selection (more spread out)
        Lower temp -> harder selection (more concentrated)
        
        Args:
            temp: New temperature value (should decrease during training)
        """
        self.temperature = temp

    def _compute_cost_matrix(self, z_, z_prior):
        """
        Compute squared L2 distance cost matrix between encoded z_ and prior samples.
        
        Args:
            z_: Encoder output of shape (B, latent_dim)
            z_prior: Prior samples of shape (N, latent_dim)
            
        Returns:
            M: Cost matrix of shape (B, N) as numpy array
        """
        # Compute squared L2 distances
        M = torch.cdist(z_, z_prior).pow(2)  # (B, N)
        
        # Normalize if needed
        if self.normalize_cost:
            M = M / (M.max() + 1e-12)
        
        return M.detach().cpu().numpy()
    
    def sinkhorn_coupling(self, z_, z_prior, use_soft=False):
        """
        Sinkhorn-based optimal transport coupling between encoded z_ and prior samples.
        
        Uses Sinkhorn algorithm to compute OT plan. Supports both hard and soft assignment modes.
        
        Hard mode (use_soft=False, default):
            - Forward pass: Hard assignment (each z_ selects one prior sample via argmax of OT plan)
            - Backward pass: Soft gradient from cost matrix (weighted combination based on distances)
            - Uses Straight-Through Estimator (STE) for gradient flow
        
        Soft mode (use_soft=True):
            - Forward and backward: Both use weighted combination from OT plan
            - More stable gradients, especially useful in early training
        
        This ensures each noise sample is matched at most once (via OT constraint).

        Args:
            z_: Encoder output of shape (B, latent_dim)
            z_prior: Prior samples of shape (N, latent_dim)
            use_soft: If True, use soft assignment (weighted combination). If False, use hard assignment with STE.

        Returns:
            z_selected: Selected latent codes of shape (B, latent_dim)
            pi: OT plan of shape (B, N) as numpy array
        """
        B, N = z_.shape[0], z_prior.shape[0]
        
        # Compute cost matrix (squared L2 distance) - keep gradient for backward pass
        M_torch = torch.cdist(z_, z_prior).pow(2)  # (B, N) - with gradient
        if self.normalize_cost:
            M_torch = M_torch / (M_torch.max() + 1e-12)
        
        # Convert to numpy for Sinkhorn
        M_np = M_torch.detach().cpu().numpy()
        
        # Uniform marginals
        a = np.ones(B) / B  # Uniform distribution over encoded z_
        b = np.ones(N) / N  # Uniform distribution over prior samples
        
        # Compute Sinkhorn OT plan
        pi = pot.sinkhorn(a, b, M_np, reg=self.sinkhorn_reg, numItermax=1000)  # (B, N)
        
        # Check for numerical errors
        if not np.all(np.isfinite(pi)):
            raise RuntimeError("Numerical error in Sinkhorn: non-finite values in OT plan")
        
        # Convert OT plan to torch tensor
        pi_tensor = torch.tensor(pi, dtype=z_.dtype, device=z_.device)  # (B, N) - no gradient
        
        if use_soft:
            # Soft mode: Use weighted combination from OT plan for both forward and backward
            z_selected = torch.matmul(pi_tensor, z_prior)  # (B, latent_dim) - weighted combination
        else:
            # Hard mode: Forward uses hard assignment, backward uses soft gradient (STE)
            # Forward: Greedy matching to ensure each prior sample is matched at most once
            # Convert to numpy for greedy matching
            pi_np = pi_tensor.detach().cpu().numpy()  # (B, N)
            
            # Greedy matching: iteratively select the best (z_enc, prior) pair
            # ensuring each prior sample is selected at most once
            indices = np.full(B, -1, dtype=np.int64)  # (B,) - which prior sample each z_ selects
            used_prior = np.zeros(N, dtype=bool)  # Track which prior samples are already used
            
            # Sort all (i, j) pairs by pi value (descending)
            pairs = []
            for i in range(B):
                for j in range(N):
                    pairs.append((pi_np[i, j], i, j))
            pairs.sort(reverse=True)  # Sort by pi value (descending)
            
            # Greedily assign matches
            for pi_val, i, j in pairs:
                if indices[i] == -1 and not used_prior[j]:
                    indices[i] = j
                    used_prior[j] = True
            
            # Handle any unmatched z_enc (shouldn't happen if B <= N, but handle edge case)
            for i in range(B):
                if indices[i] == -1:
                    # Find any unused prior sample
                    unused_j = np.where(~used_prior)[0]
                    if len(unused_j) > 0:
                        indices[i] = unused_j[0]
                        used_prior[unused_j[0]] = True
                    else:
                        # Fallback: use argmax (shouldn't happen in normal case)
                        indices[i] = np.argmax(pi_np[i])
            
            # Convert back to torch
            indices_torch = torch.from_numpy(indices).to(z_.device)  # (B,)
            z_hard = z_prior[indices_torch]  # (B, latent_dim) - actual prior samples selected
            
            # Backward: Soft gradient from cost matrix (use softmax on negative distances)
            # This approximates the OT plan gradient through the cost matrix
            weights_soft = F.softmax(-M_torch / self.sinkhorn_reg, dim=1)  # (B, N) - with gradient
            z_soft = torch.matmul(weights_soft, z_prior)  # (B, latent_dim) - weighted combination
            
            # Straight-Through Estimator: forward uses hard, backward uses soft gradient
            z_selected = z_hard.detach() + z_soft - z_soft.detach()
        
        return z_selected, pi
    
    def soft_nearest_neighbor(self, z_, z_prior):
        """
        Soft Nearest Neighbor selection from prior samples.
        
        Uses softmax on negative squared L2 distances to select prior samples.
        Supports Straight-Through Estimator (STE) for hard selection in forward
        with soft gradients in backward.
        
        Args:
            z_: Encoder output of shape (B, latent_dim)
            z_prior: Prior samples of shape (N, latent_dim)
            
        Returns:
            z_selected: Selected latent codes of shape (B, latent_dim)
            weights: Selection weights of shape (B, N)
        """
        B, N = z_.shape[0], z_prior.shape[0]
        
        # Compute squared L2 distances
        distances_sq = torch.cdist(z_, z_prior).pow(2)  # (B, N)
        
        # Normalize if needed
        if self.normalize_cost:
            distances_sq = distances_sq / (distances_sq.max() + 1e-12)
        
        # Compute softmax weights: exp(-d^2 / temperature) / sum(exp(-d^2 / temperature))
        # Negative distances for softmax (closer = higher weight)
        logits = -distances_sq / (self.temperature + 1e-12)
        weights = F.softmax(logits, dim=1)  # (B, N) - with gradient
        
        if self.use_ste:
            # Straight-Through Estimator: hard selection in forward, soft gradient in backward
            # Forward: Hard selection (argmax)
            indices = torch.argmax(weights, dim=1)  # (B,)
            z_hard = z_prior[indices]  # (B, latent_dim) - no gradient
            
            # Backward: Soft gradient from weights
            z_soft = torch.matmul(weights, z_prior)  # (B, latent_dim) - with gradient
            
            # STE: forward uses hard, backward uses soft gradient
            z_selected = z_hard.detach() + z_soft - z_soft.detach()
        else:
            # Pure soft selection: weighted combination
            z_selected = torch.matmul(weights, z_prior)  # (B, latent_dim)
        
        return z_selected, weights
    
    def ot_guided_soft_coupling(self, z_, z_prior):
        """
        OT-Guided Soft Matching: Combines 1:1 matching (Sinkhorn Hard) with soft alignment loss.
        
        Algorithm:
        1. The Matchmaker: Use Sinkhorn Hard mode to find optimal 1:1 pairs (no gradient)
        2. The Proxy: Return matched z_noise for decoder input (ensures perfect prior distribution)
        3. The Attraction: Alignment loss will be computed in loss_function using Cross Entropy
        
        Args:
            z_: Encoder output of shape (B, latent_dim)
            z_prior: Prior samples of shape (N, latent_dim)
            
        Returns:
            z_selected: Matched prior samples of shape (B, latent_dim) - no gradient
            matching_indices: Indices of matched prior samples of shape (B,) - for alignment loss
            distances_sq: Squared distances of shape (B, N) - with gradient for alignment loss
        """
        B, N = z_.shape[0], z_prior.shape[0]
        
        # Compute squared L2 distances (keep gradient for alignment loss)
        distances_sq = torch.cdist(z_, z_prior).pow(2)  # (B, N) - with gradient
        
        # Normalize if needed
        if self.normalize_cost:
            distances_sq_normalized = distances_sq / (distances_sq.max() + 1e-12)
        else:
            distances_sq_normalized = distances_sq
        
        # The Matchmaker: Use Sinkhorn Hard mode to find 1:1 matching (no gradient)
        with torch.no_grad():
            # Convert to numpy for Sinkhorn
            M_np = distances_sq_normalized.detach().cpu().numpy()
            
            # Uniform marginals
            a = np.ones(B) / B  # Uniform distribution over encoded z_
            b = np.ones(N) / N  # Uniform distribution over prior samples
            
            # Compute Sinkhorn OT plan
            pi = pot.sinkhorn(a, b, M_np, reg=self.sinkhorn_reg, numItermax=1000)  # (B, N)
            
            # Check for numerical errors
            if not np.all(np.isfinite(pi)):
                raise RuntimeError("Numerical error in Sinkhorn: non-finite values in OT plan")
            
            # Greedy matching: iteratively select the best (z_enc, prior) pair
            # ensuring each prior sample is selected at most once
            indices = np.full(B, -1, dtype=np.int64)  # (B,) - which prior sample each z_ selects
            used_prior = np.zeros(N, dtype=bool)  # Track which prior samples are already used
            
            # Sort all (i, j) pairs by pi value (descending)
            pairs = []
            for i in range(B):
                for j in range(N):
                    pairs.append((pi[i, j], i, j))
            pairs.sort(reverse=True)  # Sort by pi value (descending)
            
            # Greedily assign matches
            for pi_val, i, j in pairs:
                if indices[i] == -1 and not used_prior[j]:
                    indices[i] = j
                    used_prior[j] = True
            
            # Handle any unmatched z_enc (shouldn't happen if B <= N, but handle edge case)
            for i in range(B):
                if indices[i] == -1:
                    # Find any unused prior sample
                    unused_j = np.where(~used_prior)[0]
                    if len(unused_j) > 0:
                        indices[i] = unused_j[0]
                        used_prior[unused_j[0]] = True
                    else:
                        # Fallback: use argmax (shouldn't happen in normal case)
                        indices[i] = np.argmax(pi[i])
        
        # The Proxy: Return matched z_noise (no gradient)
        matching_indices = torch.from_numpy(indices).to(z_.device)  # (B,)
        z_selected = z_prior[matching_indices]  # (B, latent_dim) - no gradient
        
        return z_selected, matching_indices, distances_sq

    def forward(self, x, z_prior=None):
        """
        Forward pass through NO-VAE

        Args:
            x: Input data of shape (B, input_dim)
            z_prior: Prior samples of shape (N, latent_dim).
                     If None, samples from N(0, I) automatically.

        Returns:
            x_hat: Reconstructed output of shape (B, input_dim)
            z_: Encoder output of shape (B, latent_dim)
            z_selected: Selected latent codes of shape (B, latent_dim)
            weights: Selection weights of shape (B, N)
        """
        # 1. Encode: x -> z_
        z_ = self.encode(x)  # (B, latent_dim)

        # 2. Sample prior if not provided
        if z_prior is None:
            # Use batch_size if n_prior_samples is None
            n_prior = self.n_prior_samples if self.n_prior_samples is not None else x.shape[0]
            z_prior = torch.randn(
                n_prior, self.latent_dim
            ).to(x.device)

        # 3. Coupling selection
        if self.coupling_method == 'sinkhorn':
            # Determine if we should use soft mode
            # If sinkhorn_use_soft_coupling is explicitly set, use it
            # Otherwise, auto-select based on sinkhorn_reg (soft if reg > 0.01)
            if self.sinkhorn_use_soft_coupling is not None:
                use_soft = self.sinkhorn_use_soft_coupling
            else:
                use_soft = self.sinkhorn_reg > 0.01  # Auto-select: threshold for soft/hard mode
            z_selected, pi = self.sinkhorn_coupling(z_, z_prior, use_soft=use_soft)
            weights = torch.tensor(pi, dtype=z_.dtype, device=z_.device)  # Convert to tensor for compatibility
        elif self.coupling_method == 'softnn':
            z_selected, weights = self.soft_nearest_neighbor(z_, z_prior)
        elif self.coupling_method == 'ot_guided_soft':
            z_selected, matching_indices, distances_sq = self.ot_guided_soft_coupling(z_, z_prior)
            # Store matching_indices and distances_sq for alignment loss computation
            # We'll return them in a tuple with weights for compatibility
            weights = (matching_indices, distances_sq)
        else:
            raise ValueError(f"Unknown coupling method: {self.coupling_method}")

        # 4. Decode: z_selected -> x_hat
        x_hat = self.decode(z_selected)  # (B, input_dim)

        return x_hat, z_, z_selected, weights

    def loss_function(self, x, x_hat, z_, z_selected, z_recon_weight=1.0, 
                     alignment_weight=1.0, matching_info=None):
        """
        Compute total loss: reconstruction loss + latent reconstruction loss + alignment loss.
        
        Reconstruction loss: MSE between original x and reconstructed x_hat
        Latent reconstruction loss: MSE between encoded z_ and matched z_selected
          - Used for Sinkhorn and OT-Guided Soft (to align z_ with matched prior)
          - NOT used for SoftNN (original softNN only uses reconstruction loss)
        Alignment loss (for ot_guided_soft): Cross Entropy to maximize probability of matched prior
        
        Args:
            x: Original input of shape (B, input_dim)
            x_hat: Reconstructed output of shape (B, input_dim)
            z_: Encoder output of shape (B, latent_dim)
            z_selected: Selected latent codes from OT coupling of shape (B, latent_dim)
            z_recon_weight: Weight for latent reconstruction loss (default: 1.0)
              - For softnn: should be 0.0 (only reconstruction loss)
              - For sinkhorn/ot_guided_soft: can be > 0.0
            alignment_weight: Weight for alignment loss (for ot_guided_soft, default: 1.0)
            matching_info: Tuple (matching_indices, distances_sq) for ot_guided_soft method

        Returns:
            loss: Total loss (reconstruction + z_recon + alignment)
        """
        # Reconstruction loss
        loss_recon = F.mse_loss(x_hat, x, reduction='mean')
        
        # Latent reconstruction loss: encourage z_ to be close to z_selected
        # Only used for Sinkhorn and OT-Guided Soft, NOT for SoftNN
        loss_z_recon = 0.0
        if z_recon_weight > 0:
            loss_z_recon = F.mse_loss(z_selected, z_, reduction='mean')
        
        # Alignment loss (for ot_guided_soft method)
        loss_align = 0.0
        if matching_info is not None and alignment_weight > 0:
            matching_indices, distances_sq = matching_info
            # The Attraction: Cross Entropy Loss
            # Compute softmax probabilities over all prior samples
            logits = -distances_sq / (self.temperature + 1e-12)  # (B, N)
            probs = F.softmax(logits, dim=1)  # (B, N)
            
            # Cross Entropy: maximize probability of matched prior sample
            # loss_align = -log(probs[matched_index]) for each sample
            batch_indices = torch.arange(z_.shape[0], device=z_.device)  # (B,)
            matched_probs = probs[batch_indices, matching_indices]  # (B,)
            loss_align = -torch.log(matched_probs + 1e-12).mean()  # Cross Entropy
        
        # Total loss
        loss = loss_recon + z_recon_weight * loss_z_recon + alignment_weight * loss_align
        
        return loss

    def sample(self, n_samples, device='cpu'):
        """
        Generate samples by sampling z from prior and decoding (one-step).

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            x_hat: Generated samples of shape (n_samples, input_dim)
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        with torch.no_grad():
            x_hat = self.decode(z)
        return x_hat


if __name__ == "__main__":
    # Test NO-VAE
    print("Testing NO-VAE with Sinkhorn coupling...")

    # Create model
    model = NOVAE(input_dim=1, latent_dim=1, n_prior_samples=256, coupling_method='sinkhorn')
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1)
    z_prior = torch.randn(256, 1)
    x_hat, z_, z_selected, weights = model(x, z_prior)
    print(f"\nInput shape: {x.shape}")
    print(f"Encoder output z_ shape: {z_.shape}")
    print(f"Selected z shape: {z_selected.shape}")
    print(f"Weights (OT plan) shape: {weights.shape}")
    print(f"Output shape: {x_hat.shape}")

    # Test loss computation
    loss = model.loss_function(x, x_hat)
    print(f"\nReconstruction loss: {loss.item():.4f}")

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"\nSample shape: {samples.shape}")

    # Test 2D
    model_2d = NOVAE(input_dim=2, latent_dim=2, n_prior_samples=256, coupling_method='sinkhorn')
    x_2d = torch.randn(10, 2)
    x_hat_2d, z_2d, z_sel_2d, w_2d = model_2d(x_2d)
    print(f"\n2D - Input: {x_2d.shape}, Output: {x_hat_2d.shape}")
    print(f"2D - z_: {z_2d.shape}, z_selected: {z_sel_2d.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters (1D): {n_params}")
