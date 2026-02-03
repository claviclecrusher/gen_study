import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backflow import BackFlow
from data.synthetic import sample_prior
from visualization.viz_backflow_2d import compute_trajectories_2d

# Create a simple model
model = BackFlow(input_dim=2).cuda()

# Test with a few samples
z = torch.FloatTensor(sample_prior(n_samples=5, seed=42, dim=2)).cuda()
print("Initial z:")
print(z.cpu().numpy())

# Compute trajectories
trajectories = compute_trajectories_2d(model, z, steps=10)
print(f"\nTrajectory shape: {trajectories.shape}")
print("\nTrajectories at each step:")
for step in range(trajectories.shape[0]):
    print(f"Step {step}: {trajectories[step, 0, :]}")  # Print first sample trajectory

# Check if velocities are being computed
print("\n\nChecking velocity predictions:")
with torch.no_grad():
    t = torch.ones(5).cuda()
    v = model(z, t, t)
    print(f"Velocity at t=1: {v.cpu().numpy()}")
    
print("\n\nChecking if trajectory points are different:")
for i in range(min(5, trajectories.shape[0]-1)):
    diff = np.linalg.norm(trajectories[i+1, 0, :] - trajectories[i, 0, :])
    print(f"Distance from step {i} to {i+1}: {diff}")
