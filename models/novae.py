"""
Noise Oriented VAE (NO-VAE) model

Key idea: Replace VAE's Gaussian reparameterization with bridging between
encoded z_ and prior samples. The encoder outputs a single deterministic point
z_ instead of (mu, logvar). Among N prior samples, one is selected via bridging
(Sinkhorn OT or Soft Nearest Neighbor). The selected sample is then decoded
for reconstruction.

This naturally aligns the encoder output distribution with the prior without
explicit KL divergence or any regularization loss.

Bridging methods (z_enc <-> z_noise matching; distinct from CFM coupling):
- 'sinkhorn': Optimal Transport bridging via Sinkhorn algorithm (ensures each
  noise sample is matched at most once)
- 'softnn': Soft Nearest Neighbor selection using softmax on negative squared
  L2 distances (allows multiple z_enc to select the same prior sample)
- 'inv_softnn': Inverted SoftNN (Noise-to-Encoder Selection). Each noise selects
  exactly one encoder (blocks Many-to-One). One encoder can receive multiple noises.
  Load balancing and distance minimization via auxiliary losses.
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
        Selection: z_ -> z_selected (OT bridging via Sinkhorn from z_prior)
        Decoding: z_selected -> x_hat
        Loss: MSE(x, x_hat) (reconstruction only, no regularization)

    Inference (Generation):
        Sample z ~ N(0, I) -> Decode -> x_hat (one-step, no ODE)

    The decoder always receives input that is a prior sample (via OT bridging),
    so it naturally learns to map from the prior distribution to the data distribution.
    OT bridging ensures each noise sample is matched at most once.

    Args:
        input_dim: Dimension of input/output space (default: 1)
        latent_dim: Dimension of latent space (default: 1)
        hidden_dims: Hidden layer dimensions (default: [32, 64, 32])
        n_prior_samples: Number of prior samples per mini-batch (default: None, uses batch_size or n_samples depending on context)
        bridging_method: Bridging method ('sinkhorn', 'softnn', or 'ot_guided_soft', default: 'sinkhorn')
        sinkhorn_reg: Entropic regularization for Sinkhorn (default: 0.05)
        normalize_cost: Whether to normalize cost matrix (default: True)
        sinkhorn_use_soft_bridging: For Sinkhorn, whether to use soft bridging (default: False)
        temperature: Temperature for Soft NN (default: 0.1)
        use_ste: For Soft NN, whether to use Straight-Through Estimator (default: False)
    """

    def __init__(self, input_dim=1, latent_dim=1, hidden_dims=None,
                 n_prior_samples=None, bridging_method='sinkhorn',
                 coupling_method=None,  # deprecated, use bridging_method
                 sinkhorn_reg=0.05, normalize_cost=True, sinkhorn_use_soft_bridging=False,
                 sinkhorn_use_soft_coupling=None,  # deprecated, use sinkhorn_use_soft_bridging
                 temperature=0.1, use_ste=False):
        super(NOVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 32]

        # Backward compatibility
        if coupling_method is not None:
            bridging_method = coupling_method
        if sinkhorn_use_soft_coupling is not None:
            sinkhorn_use_soft_bridging = sinkhorn_use_soft_coupling

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_prior_samples = n_prior_samples
        self.bridging_method = bridging_method
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_reg_init = sinkhorn_reg  # Store initial value for annealing
        self.normalize_cost = normalize_cost
        # If None, auto-select based on sinkhorn_reg (soft if reg > 0.01)
        self.sinkhorn_use_soft_bridging = sinkhorn_use_soft_bridging
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
        
        Higher reg -> softer bridging (more spread out OT plan)
        Lower reg -> harder bridging (more concentrated OT plan)
        
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
    
    def sinkhorn_bridging(self, z_, z_prior, use_soft=False):
        """
        Sinkhorn-based optimal transport bridging between encoded z_ and prior samples.
        
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

    def inverted_softnn_bridging(self, z_, z_prior):
        """
        Inverted Soft Nearest Neighbor (Noise-to-Encoder Selection)

        Satisfies user constraints:
        1) No Many-to-One: Each noise selects exactly one encoder (via argmax dim=0).
        2) Allow One-to-Many: One encoder can be selected by multiple noises.
        3) Balanced Load: Addressed via auxiliary loss (std of soft counts).
        4) Min Distance: Encoders pick the closest noise from their assigned partition.

        Args:
            z_: (B, D) - encoder outputs
            z_prior: (N, D) - prior samples (noise)

        Returns:
            z_selected: (B, D)
            info: dict containing 'counts', 'distances_sq', 'selected_indices' for loss
        """
        B, N = z_.shape[0], z_prior.shape[0]

        # 1. Compute Distances
        # distances_sq: (B, N) - distances_sq[i,j] = ||z_[i] - z_prior[j]||^2
        distances_sq = torch.cdist(z_, z_prior).pow(2)

        if self.normalize_cost:
            scale = distances_sq.detach().max() + 1e-12
            dists_norm = distances_sq / scale
        else:
            dists_norm = distances_sq

        # 2. Inverted Softmax (Dim=0: Noise chooses Encoder)
        # Prob(Noise j belongs to Encoder i) - sum over encoders (dim=0) is 1 per noise
        logits = -dists_norm / (self.temperature + 1e-12)
        probs_inverted = F.softmax(logits, dim=0)  # (B, N), sum over B is 1 per column

        # 3. Hard Assignment (Forward Pass)
        # Each noise picks its owner (Constraint 1 satisfied)
        with torch.no_grad():
            owners = probs_inverted.argmax(dim=0)  # (N,) values in 0..B-1

            # Count how many noises each encoder got (for Loss Constraint 3)
            counts = torch.bincount(owners, minlength=B).float().to(z_.device)  # (B,)

            # 4. Selection within Partition (Constraint 4 satisfied)
            # For each encoder, pick the BEST (closest) noise among those that selected it.
            selected_indices = torch.zeros(B, dtype=torch.long, device=z_.device)

            for i in range(B):
                my_noises = torch.where(owners == i)[0]

                if len(my_noises) > 0:
                    best_local_idx = distances_sq[i, my_noises].argmin()
                    selected_indices[i] = my_noises[best_local_idx]
                else:
                    # Fallback: If no noise chose this encoder (e.g. outlier z_enc),
                    # pick the globally closest noise
                    selected_indices[i] = distances_sq[i].argmin()

        # 5. Backward Pass (STE)
        z_hard = z_prior[selected_indices]  # (B, D)

        # Gradient Trick: z_soft = weighted average of noises per encoder
        row_sums = probs_inverted.sum(dim=1, keepdim=True) + 1e-12
        probs_row_norm = probs_inverted / row_sums
        z_soft = torch.matmul(probs_row_norm, z_prior)

        z_selected = z_hard.detach() + z_soft - z_soft.detach()

        info = {
            'counts': counts,
            'distances_sq': distances_sq,
            'selected_indices': selected_indices
        }

        return z_selected, info

    def ot_guided_soft_bridging(self, z_, z_prior):
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

    def nep_bridging(self, z_, z_prior, use_soft=False):
        """
        Neural Equi-Partition (NEP) bridging.

        Partitions N noise samples among B latent codes via rectangular Sinkhorn OT
        so that each z_enc gets ~N/B noise samples (equi-partition).
        Then randomly selects one noise from each partition for decoding.

        Marginals:
            row (z_enc side):  a_i = N/B  (each z_enc gets equal mass)
            column (noise side): b_j = 1  (each noise assigned to exactly one z_enc)

        Args:
            z_: Encoder output of shape (B, latent_dim)
            z_prior: Prior samples of shape (N, latent_dim)
            use_soft: If False (default), use STE (forward hard, backward soft).
                      If True, use soft P* row-normalized weights for weighted sum.

        Returns:
            z_selected: Selected noise of shape (B, latent_dim)
            nep_info: Dict with 'assignments' (N,), 'distances_sq' (B, N) for NEP loss
        """
        B, N = z_.shape[0], z_prior.shape[0]

        # 1. Compute squared L2 distances (keep gradient for NEP loss)
        distances_sq = torch.cdist(z_, z_prior).pow(2)  # (B, N) - with gradient

        # Normalized cost for Sinkhorn numerical stability
        if self.normalize_cost:
            M_normalized = distances_sq.detach() / (distances_sq.detach().max() + 1e-12)
        else:
            M_normalized = distances_sq.detach()
        M_np = M_normalized.cpu().numpy()

        # 2. Marginals (normalized to sum to 1 for POT convention)
        # Actual marginals: a_i = N/B, b_j = 1 (total mass = N)
        # Normalized: a_i = 1/B, b_j = 1/N (total mass = 1)
        a = np.ones(B) / B
        b = np.ones(N) / N

        # 3. Rectangular Sinkhorn OT
        pi = pot.sinkhorn(a, b, M_np, reg=self.sinkhorn_reg, numItermax=1000)  # (B, N)
        if not np.all(np.isfinite(pi)):
            raise RuntimeError("Numerical error in NEP Sinkhorn: non-finite values in OT plan")

        pi_tensor = torch.tensor(pi, dtype=z_.dtype, device=z_.device)  # (B, N)

        # 4. Hard assignment: for each noise j, assign to argmax_i P_ij
        with torch.no_grad():
            assignments = pi_tensor.argmax(dim=0)  # (N,) - which z_i each noise j belongs to

            # Balanced redistribution: ensure every z_enc gets at least 1 noise.
            # argmax discretization can leave many partitions empty while a few
            # hoard all noises. We steal noises from overfull (>1) partitions
            # and give them to empty ones, preferring noises with high P* for
            # the receiving z_enc.
            counts = torch.bincount(assignments, minlength=B)
            while (counts == 0).any() and (counts > 1).any():
                emp_i = (counts == 0).nonzero(as_tuple=True)[0][0].item()
                # Stealable noises: those in partitions with count > 1
                stealable = (counts[assignments] > 1)  # (N,) bool mask
                candidate_j = stealable.nonzero(as_tuple=True)[0]
                # Pick the candidate with highest Sinkhorn probability for emp_i
                best_j = candidate_j[pi_tensor[emp_i, candidate_j].argmax()].item()
                old_owner = assignments[best_j].item()
                assignments[best_j] = emp_i
                counts[emp_i] += 1
                counts[old_owner] -= 1

            # Partition mask: M_mask[i, j] = 0 if noise j assigned to enc i, else -inf
            # This ensures each noise belongs to exactly one z_enc's partition
            M_mask = torch.full((B, N), float('-inf'), device=z_.device)
            noise_indices = torch.arange(N, device=z_.device)
            M_mask[assignments, noise_indices] = 0.0

            # 5. For each z_i, randomly select one noise from its partition
            selected_indices = torch.zeros(B, dtype=torch.long, device=z_.device)
            for i in range(B):
                partition_indices = torch.where(assignments == i)[0]
                if len(partition_indices) > 0:
                    rand_idx = partition_indices[torch.randint(len(partition_indices), (1,))]
                    selected_indices[i] = rand_idx
                else:
                    # This should not happen after redistribution, but just in case
                    # (only possible if N < B)
                    selected_indices[i] = pi_tensor[i].argmax()

        if use_soft:
            # Soft mode: row-normalized P* as weighted sum (forward and backward)
            # Apply partition mask so each z_enc only attends to its own partition
            masked_pi = pi_tensor * (M_mask == 0).float()
            row_sums = masked_pi.sum(dim=1, keepdim=True) + 1e-12
            pi_row_norm = masked_pi / row_sums  # (B, N) row-normalized within partition
            z_selected = torch.matmul(pi_row_norm, z_prior)  # (B, latent_dim)
        else:
            # STE mode: forward hard (random from partition), backward soft
            z_hard = z_prior[selected_indices]  # (B, latent_dim) - no gradient

            # Masked softmax: only consider noise within own partition
            # Noise outside partition gets -inf logit -> 0 probability -> no gradient
            masked_logits = -distances_sq / (self.sinkhorn_reg + 1e-12) + M_mask
            weights_soft = F.softmax(masked_logits, dim=1)  # (B, N)
            z_soft = torch.matmul(weights_soft, z_prior)  # (B, latent_dim) - with gradient

            # Straight-Through Estimator
            z_selected = z_hard.detach() + z_soft - z_soft.detach()

        nep_info = {
            'assignments': assignments,      # (N,) hard partition assignments
            'distances_sq': distances_sq,    # (B, N) raw squared distances with gradient
        }

        return z_selected, nep_info

    def forward(self, x, z_prior=None, no_sampling_ratio=None):
        """
        Forward pass through NO-VAE

        Args:
            x: Input data of shape (B, input_dim)
            z_prior: Prior samples of shape (N, latent_dim).
                     If None, samples from N(0, I) automatically.
            no_sampling_ratio: Float in [0, 1]. For this fraction of the batch,
                use z_enc directly in the decoder (no bridging with noise) for
                reconstruction. If None or 0, use bridging for all samples.

        Returns:
            x_hat: Reconstructed output of shape (B, input_dim)
            z_: Encoder output of shape (B, latent_dim)
            z_selected: Selected latent codes of shape (B, latent_dim)
            weights: Selection weights of shape (B, N)
            no_sampling_mask: Boolean mask of shape (B,) - True where z_enc was
                used directly. None if no_sampling_ratio is None or 0.
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

        # 3. Bridging selection
        if self.bridging_method == 'sinkhorn':
            # Determine if we should use soft mode
            # If sinkhorn_use_soft_bridging is explicitly set, use it
            # Otherwise, auto-select based on sinkhorn_reg (soft if reg > 0.01)
            if self.sinkhorn_use_soft_bridging is not None:
                use_soft = self.sinkhorn_use_soft_bridging
            else:
                use_soft = self.sinkhorn_reg > 0.01  # Auto-select: threshold for soft/hard mode
            z_selected, pi = self.sinkhorn_bridging(z_, z_prior, use_soft=use_soft)
            weights = torch.tensor(pi, dtype=z_.dtype, device=z_.device)  # Convert to tensor for compatibility
        elif self.bridging_method == 'softnn':
            z_selected, weights = self.soft_nearest_neighbor(z_, z_prior)
        elif self.bridging_method == 'inv_softnn':
            z_selected, inv_softnn_info = self.inverted_softnn_bridging(z_, z_prior)
            weights = inv_softnn_info
        elif self.bridging_method == 'ot_guided_soft':
            z_selected, matching_indices, distances_sq = self.ot_guided_soft_bridging(z_, z_prior)
            # Store matching_indices and distances_sq for alignment loss computation
            # We'll return them in a tuple with weights for compatibility
            weights = (matching_indices, distances_sq)
        elif self.bridging_method == 'nep':
            # Neural Equi-Partition: partition N noise among B z_enc
            if self.sinkhorn_use_soft_bridging is not None:
                use_soft = self.sinkhorn_use_soft_bridging
            else:
                use_soft = self.sinkhorn_reg > 0.01
            z_selected, nep_info = self.nep_bridging(z_, z_prior, use_soft=use_soft)
            weights = nep_info  # dict with 'assignments' and 'distances_sq'
        else:
            raise ValueError(f"Unknown bridging method: {self.bridging_method}")

        # 4. no_sampling_ratio: use z_enc directly for decode (no bridging) for a fraction of the batch
        no_sampling_mask = None
        if no_sampling_ratio is not None and no_sampling_ratio > 0 and self.training:
            B = z_.shape[0]
            n_no_sampling = max(1, int(no_sampling_ratio * B))
            # Randomly select which samples use z_enc directly
            perm = torch.randperm(B, device=z_.device)
            no_sampling_indices = perm[:n_no_sampling]
            no_sampling_mask = torch.zeros(B, dtype=torch.bool, device=z_.device)
            no_sampling_mask[no_sampling_indices] = True
            z_for_decode = torch.where(no_sampling_mask.unsqueeze(1), z_, z_selected)
        else:
            z_for_decode = z_selected

        # 5. Decode: z_for_decode -> x_hat
        x_hat = self.decode(z_for_decode)  # (B, input_dim)

        return x_hat, z_, z_selected, weights, no_sampling_mask

    def loss_function(self, x, x_hat, z_, z_selected, z_recon_weight=0.0,
                     alignment_weight=1.0, matching_info=None, no_sampling_mask=None,
                     beta=0.0, nep_weight=0.0, nep_var_weight=0.0, nep_info=None,
                     inv_softnn_info=None, count_var_weight=0.0):
        """
        Compute total loss: reconstruction + latent recon + alignment + reg + NEP loss.
        
        Reconstruction loss: MSE between original x and reconstructed x_hat
        Latent reconstruction loss: MSE between encoded z_ and matched z_selected
          - Used for Sinkhorn and OT-Guided Soft (to align z_ with matched prior)
          - NOT used for SoftNN, NEP (they have their own alignment mechanisms)
          - NOT used for samples with no_sampling_mask=True (z_enc used directly)
        Alignment loss (for ot_guided_soft): Cross Entropy to maximize probability of matched prior
          - NOT used for samples with no_sampling_mask=True
        Regularization loss (beta > 0): Sample-based KL divergence. Computes batch mean and variance
          of z_enc, then KL(N(batch_mean, batch_var) || N(0,I)). Unlike VAE, measured at batch level.
        NEP loss (nep_weight > 0): L_NEP = (1/N) * sum_j ||z_{sigma_j} - e_j||^2
          - Encourages z_enc to be close to its assigned noise partition
        NEP variance penalty (nep_var_weight > 0): std of per-sample ||z_enc_i - z_selected_i||^2
          - Penalizes uneven distances: some z_enc close, others far from their noise
        
        Args:
            x: Original input of shape (B, input_dim)
            x_hat: Reconstructed output of shape (B, input_dim)
            z_: Encoder output of shape (B, latent_dim)
            z_selected: Selected latent codes from OT bridging of shape (B, latent_dim)
            z_recon_weight: Weight for latent reconstruction loss (default: 1.0)
              - For softnn/nep: should be 0.0
              - For sinkhorn/ot_guided_soft: can be > 0.0
            alignment_weight: Weight for alignment loss (for ot_guided_soft, default: 1.0)
            matching_info: Tuple (matching_indices, distances_sq) for ot_guided_soft method
            no_sampling_mask: Boolean mask (B,) - True where z_enc was used directly.
                z_recon and alignment losses are only applied where mask is False.
            beta: Weight for regularization loss (default: 0.0). Sample-based KL(z_enc batch || N(0,I)).
            nep_weight: Weight for NEP loss (default: 0.0).
            nep_var_weight: Weight for NEP distance variance penalty (default: 0.0).
                Penalizes std of per-sample squared distances between z_enc and z_selected.
            nep_info: Dict with 'assignments' (N,) and 'distances_sq' (B, N) from nep_bridging.
            inv_softnn_info: Dict with 'counts', 'distances_sq' from inverted_softnn_bridging.
            count_var_weight: Weight for Inverted SoftNN load balancing loss (std of soft counts).

        Returns:
            loss: Total loss (reconstruction + z_recon + alignment + beta * reg
                  + nep_weight * nep + nep_var_weight * nep_var)
        """
        # Reconstruction loss
        loss_recon = F.mse_loss(x_hat, x, reduction='mean')
        
        # Latent reconstruction loss: encourage z_ to be close to z_selected
        # Only used for Sinkhorn and OT-Guided Soft, NOT for SoftNN, NOT for no_sampling samples
        loss_z_recon = 0.0
        if z_recon_weight > 0:
            if no_sampling_mask is not None and no_sampling_mask.any():
                # Only apply to samples that used bridging (mask False)
                coupled_mask = ~no_sampling_mask
                if coupled_mask.any():
                    loss_z_recon = F.mse_loss(
                        z_selected[coupled_mask], z_[coupled_mask], reduction='mean'
                    )
            else:
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
            if no_sampling_mask is not None and no_sampling_mask.any():
                # Only apply to bridged samples
                coupled_mask = ~no_sampling_mask
                if coupled_mask.any():
                    loss_align = -torch.log(matched_probs[coupled_mask] + 1e-12).mean()
            else:
                loss_align = -torch.log(matched_probs + 1e-12).mean()  # Cross Entropy
        
        # Regularization loss: sample-based KL(z_enc batch || N(0,I))
        # z_enc batch mean and variance over batch dimension
        loss_reg = 0.0
        if beta > 0:
            z_mean = z_.mean(dim=0)  # (latent_dim,)
            z_var = z_.var(dim=0, unbiased=True) + 1e-8  # (latent_dim,), avoid log(0)
            # KL(N(mean, var) || N(0, 1)) per dim: 0.5 * (mean^2 + var - 1 - log(var))
            kl_per_dim = 0.5 * (z_mean.pow(2) + z_var - 1.0 - z_var.log())
            loss_reg = kl_per_dim.sum()
        
        # NEP loss: L_NEP = (1/N) * sum_j ||z_{sigma_j} - e_j||^2
        loss_nep = 0.0
        if nep_info is not None and nep_weight > 0:
            assignments = nep_info['assignments']      # (N,) which z_i each noise j is assigned to
            distances_sq = nep_info['distances_sq']    # (B, N) raw squared distances with gradient
            N = distances_sq.shape[1]
            j_indices = torch.arange(N, device=z_.device)
            # For each noise j, get distance to its assigned z_enc
            assigned_distances = distances_sq[assignments, j_indices]  # (N,)
            loss_nep = assigned_distances.mean()
        
        # NEP distance variance penalty: penalize uneven per-sample distances
        # Encourages all z_enc to be similarly distant from their matched z_selected
        loss_nep_var = 0.0
        if nep_var_weight > 0:
            dists_sq_per_sample = ((z_ - z_selected) ** 2).sum(dim=1)  # (B,)
            loss_nep_var = dists_sq_per_sample.std()

        # Inverted SoftNN specific: Load balancing loss (Constraint 3)
        # Distance minimization (Constraint 4) is handled by loss_z_recon (z_recon_weight)
        loss_count_balance = 0.0
        if inv_softnn_info is not None and count_var_weight > 0:
            dists = inv_softnn_info['distances_sq']
            if self.normalize_cost:
                scale = dists.detach().max() + 1e-12
                dists_norm = dists / scale
            else:
                dists_norm = dists
            logits = -dists_norm / (self.temperature + 1e-12)
            probs = F.softmax(logits, dim=0)  # (B, N)
            soft_counts = probs.sum(dim=1)  # (B,) - differentiable
            loss_count_balance = soft_counts.std()

        # Total loss
        loss = (loss_recon + z_recon_weight * loss_z_recon + alignment_weight * loss_align
                + beta * loss_reg + nep_weight * loss_nep
                + nep_var_weight * loss_nep_var
                + count_var_weight * loss_count_balance)
        
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
    print("Testing NO-VAE with Sinkhorn bridging...")

    # Create model
    model = NOVAE(input_dim=1, latent_dim=1, n_prior_samples=256, bridging_method='sinkhorn')
    print(f"Model architecture:\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.randn(batch_size, 1)
    z_prior = torch.randn(256, 1)
    x_hat, z_, z_selected, weights, _ = model(x, z_prior)
    print(f"\nInput shape: {x.shape}")
    print(f"Encoder output z_ shape: {z_.shape}")
    print(f"Selected z shape: {z_selected.shape}")
    print(f"Weights (OT plan) shape: {weights.shape}")
    print(f"Output shape: {x_hat.shape}")

    # Test loss computation
    loss = model.loss_function(x, x_hat, z_, z_selected)
    print(f"\nReconstruction loss: {loss.item():.4f}")

    # Test sampling
    samples = model.sample(n_samples=100)
    print(f"\nSample shape: {samples.shape}")

    # Test 2D
    model_2d = NOVAE(input_dim=2, latent_dim=2, n_prior_samples=256, bridging_method='sinkhorn')
    x_2d = torch.randn(10, 2)
    x_hat_2d, z_2d, z_sel_2d, w_2d, _ = model_2d(x_2d)
    print(f"\n2D - Input: {x_2d.shape}, Output: {x_hat_2d.shape}")
    print(f"2D - z_: {z_2d.shape}, z_selected: {z_sel_2d.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters (1D): {n_params}")
