"""
Test BackFlow 2D training with more epochs to verify velocity learning
"""
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backflow import BackFlow
from data.synthetic import generate_data_2d, sample_prior
from visualization.viz_backflow_2d import compute_trajectories_2d
from training.train_backflow import compute_imf_loss
import torch.optim as optim

# Set seed
torch.manual_seed(42)
np.random.seed(42)

# Create model and data
model = BackFlow(input_dim=2).cuda()
x_data = generate_data_2d(n_samples=500, seed=42)
x_tensor = torch.FloatTensor(x_data).cuda()

# Create dataloader
dataset = torch.utils.data.TensorDataset(x_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

# Train for 200 epochs
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print('Training BackFlow 2D for 200 epochs...')

for epoch in range(200):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for (batch_x,) in dataloader:
        # Use the correct loss function from train_backflow.py
        loss = compute_imf_loss(model, batch_x)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (same as original)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    epoch_loss /= n_batches

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1}/200, Loss: {epoch_loss:.6f}')

# Test trajectories after training
print('\nTesting trajectories after 200 epochs of training:')
model.eval()
z = torch.FloatTensor(sample_prior(n_samples=5, seed=42, dim=2)).cuda()

with torch.no_grad():
    trajectories = compute_trajectories_2d(model, z, steps=10)

    # Check velocities
    t = torch.ones(5).cuda()
    v = model(z, t, t)
    print(f'\nVelocity at t=1:\n{v.cpu().numpy()}')

    print('\nTrajectory distances (should be non-zero if model learned):')
    for i in range(min(5, trajectories.shape[0]-1)):
        diff = np.linalg.norm(trajectories[i+1, 0, :] - trajectories[i, 0, :])
        print(f'Distance from step {i} to {i+1}: {diff:.6f}')

    print('\nFirst sample trajectory:')
    print(f'Start (t=1): {trajectories[0, 0, :]}')
    print(f'End (t=0): {trajectories[-1, 0, :]}')
    print(f'Total displacement: {np.linalg.norm(trajectories[-1, 0, :] - trajectories[0, 0, :]):.6f}')
