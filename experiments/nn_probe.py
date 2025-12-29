"""
Neural Network Topological Probe

Trains an MLP to learn the N=10 Parity function while monitoring
the topology of the learned truth table using PCC.

Phase 3: Dynamic topology tracking during gradient descent.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
import ripser

# Configuration
N = 10
HIDDEN_SIZE = 128
EPOCHS = 500
PROBE_INTERVAL = 10
LR = 0.01

def get_hypercube_data(n):
    """Generate all 2^n binary vectors and parity labels."""
    indices = np.arange(2**n, dtype=int)
    masks = 1 << np.arange(n)[::-1]
    X = ((indices[:, None] & masks) > 0).astype(np.float32)
    
    # Parity = 1 if odd number of 1s
    y = (X.sum(axis=1) % 2).astype(np.float32)
    
    return X, y

def add_jitter(points, sigma=0.05):
    """Add Gaussian jitter."""
    return points + np.random.normal(0, sigma, points.shape)

def compute_pcc(coords, max_dim=1):
    """Compute PCC (Total Persistence) for H1+."""
    if len(coords) < 3:
        return 0.0
    
    # Sample if too large
    if len(coords) > 500:
        indices = np.random.choice(len(coords), 500, replace=False)
        coords = coords[indices]
    
    # Jitter and compute distance
    pts = add_jitter(coords.astype(float), sigma=0.05)
    dist_matrix = squareform(pdist(pts, metric='euclidean'))
    
    # Ripser
    results = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=max_dim)
    dgms = results['dgms']
    
    # Total persistence for H1+
    total = 0.0
    for dim in range(1, len(dgms)):
        dgm = dgms[dim]
        if len(dgm) == 0:
            continue
        finite = dgm[~np.isinf(dgm[:, 1])]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            total += np.sum(lifetimes ** 2)
    
    return total

class ParityMLP(nn.Module):
    """MLP for learning parity function."""
    def __init__(self, input_size=10, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

def train_with_topology_probe():
    print("=" * 60)
    print("NEURAL NETWORK TOPOLOGICAL PROBE")
    print("=" * 60)
    print(f"\nConfig: N={N}, Hidden={HIDDEN_SIZE}, Epochs={EPOCHS}, LR={LR}")
    
    # Generate data
    X_np, y_np = get_hypercube_data(N)
    X = torch.tensor(X_np)
    y = torch.tensor(y_np)
    
    print(f"Data: {len(X)} samples ({int(y.sum().item())} positive)")
    
    # Model
    model = ParityMLP(input_size=N, hidden_size=HIDDEN_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    # Logging
    logs = []
    
    print(f"\n{'Epoch':<8} | {'Loss':<10} | {'Accuracy':<10} | {'PCC':<12}")
    print("-" * 50)
    
    for epoch in range(1, EPOCHS + 1):
        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Probe every N epochs
        if epoch % PROBE_INTERVAL == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds = model(X)
                binary_preds = (preds > 0.5).float()
                accuracy = (binary_preds == y).float().mean().item()
                
                # Get "learned truth table" - coordinates where model predicts 1
                on_set_mask = binary_preds.numpy() == 1
                on_set_coords = X_np[on_set_mask]
                
                # Compute PCC
                pcc = compute_pcc(on_set_coords) if len(on_set_coords) > 2 else 0.0
            
            logs.append({
                'epoch': epoch,
                'loss': loss.item(),
                'accuracy': accuracy,
                'pcc': pcc,
                'on_set_size': len(on_set_coords)
            })
            
            print(f"{epoch:<8} | {loss.item():<10.4f} | {accuracy:<10.4f} | {pcc:<12.2f}")
    
    # Save logs
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_topology.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy', 'pcc', 'on_set_size'])
        writer.writeheader()
        writer.writerows(logs)
    
    print(f"\nLogs saved to: {output_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    final = logs[-1]
    print(f"\nFinal Accuracy: {final['accuracy']:.4f}")
    print(f"Final PCC: {final['pcc']:.2f}")
    
    # Topology trajectory
    print("\nPCC Trajectory:")
    trajectory = [(l['epoch'], l['pcc']) for l in logs[::5]]  # Every 5th entry
    for epoch, pcc in trajectory:
        print(f"  Epoch {epoch:>4}: PCC = {pcc:.2f}")

if __name__ == "__main__":
    train_with_topology_probe()
